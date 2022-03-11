import os
import shutil
import time
from pathlib import Path
from typing import DefaultDict, Dict, Union

import numpy as np
import torch
from config import config, global_params
from tqdm.auto import tqdm

from src import callbacks, metrics, models, utils

LOGS_PARAMS = global_params.LogsParams()

training_logger = config.init_logger(
    log_file=Path.joinpath(LOGS_PARAMS.LOGS_DIR_RUN_ID, "training.log"),
    module_name="training",
)


def get_sigmoid_softmax(
    pipeline_config,
) -> Union[torch.nn.Sigmoid, torch.nn.Softmax]:
    """Get the sigmoid or softmax function.

    Returns:
        Union[torch.nn.Sigmoid, torch.nn.Softmax]: [description]
    """
    if (
        pipeline_config.criterion_params.train_criterion_name
        == "BCEWithLogitsLoss"
    ):
        return getattr(torch.nn, "Sigmoid")()

    if (
        pipeline_config.criterion_params.train_criterion_name
        == "CrossEntropyLoss"
    ):
        return getattr(torch.nn, "Softmax")(dim=1)


class Trainer:
    """Object used to facilitate training."""

    def __init__(
        self,
        pipeline_config: global_params.PipelineConfig,
        model: models.CustomNeuralNet,
        model_artifacts_path: Union[str, Path],
        device=torch.device("cpu"),
        wandb_run=None,
        early_stopping: callbacks.EarlyStopping = None,
    ):
        # Set params
        self.pipeline_config = pipeline_config
        self.params = self.pipeline_config.global_train_params
        self.model = model
        self.model_path = model_artifacts_path
        self.device = device

        self.wandb_run = wandb_run
        self.early_stopping = early_stopping

        # TODO: To ask Ian if initializing the optimizer in constructor is a good idea? Should we init it outside of the class like most people do? In particular, the memory usage.
        self.optimizer = self.get_optimizer(
            model=self.model,
            optimizer_params=self.pipeline_config.optimizer_params,
        )
        self.scheduler = self.get_scheduler(
            optimizer=self.optimizer,
            scheduler_params=self.pipeline_config.scheduler_params,
        )

        if self.params.use_amp:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # list to contain various train metrics
        # TODO: how to add more metrics? wandb log too. Maybe save to model artifacts?
        self.monitored_metric = {
            "metric_name": "valid_macro_auroc",
            "metric_score": None,
            "mode": "max",
        }
        # Metric to optimize, either min or max.
        self.best_valid_score = (
            -np.inf if self.monitored_metric["mode"] == "max" else np.inf
        )
        self.patience_counter = self.params.patience  # Early Stopping Counter
        self.history = DefaultDict(list)

    @staticmethod
    def get_optimizer(
        model: models.CustomNeuralNet,
        optimizer_params,
    ):
        """Get the optimizer for the model.

        Caution: Do not invoke self.model directly in this call as it may affect model initalization.
        READ: https://stackoverflow.com/questions/70107044/can-i-define-a-method-as-an-attribute

        Args:
            model (models.CustomNeuralNet): The model to optimize.
            optimizer_params (Dict[str, Any]): The parameters for the optimizer.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer for the model.
        """
        return getattr(torch.optim, optimizer_params.optimizer_name)(
            model.parameters(), **optimizer_params.optimizer_params
        )

    @staticmethod
    def get_scheduler(
        optimizer: torch.optim,
        scheduler_params,
    ):
        """Get the scheduler for the optimizer.

        Args:
            optimizer (torch.optim): The optimizer to use.
            scheduler_params (global_params.SchedulerParams): The scheduler parameters.

        Returns:
            scheduler (torch.optim.lr_scheduler): The scheduler.
        """

        return getattr(
            torch.optim.lr_scheduler, scheduler_params.scheduler_name
        )(optimizer=optimizer, **scheduler_params.scheduler_params)

    @staticmethod
    def train_criterion(
        y_true: torch.Tensor,
        y_logits: torch.Tensor,
        batch_size: int,
        criterion_params,
    ):
        """Train Loss Function.
        Note that we can evaluate train and validation fold with different loss functions.

        The below example applies for CrossEntropyLoss.

        Args:
            y_true ([type]): Input - N,C) where N = number of samples and C = number of classes.
            y_logits ([type]): If containing class indices, shape (N) where each value is 0 \leq \text{targets}[i] \leq C-10≤targets[i]≤C−1
                               If containing class probabilities, same shape as the input.
            criterion_params (global_params.CriterionParams, optional): [description]. Defaults to CRITERION_PARAMS.criterion_params.

        Returns:
            [type]: [description]
        """

        loss_fn = getattr(torch.nn, criterion_params.train_criterion_name)(
            **criterion_params.train_criterion_params
        )

        if criterion_params.train_criterion_name == "CrossEntropyLoss":
            pass
        elif criterion_params.train_criterion_name == "BCEWithLogitsLoss":
            assert (
                y_logits.shape[0] == y_true.shape[0] == batch_size
            ), f"BCEWithLogitsLoss expects first dimension to be batch size {batch_size}"
            assert (
                y_logits.shape == y_true.shape
            ), "BCEWithLogitsLoss inputs must be of the same shape."

        loss = loss_fn(y_logits, y_true)
        return loss

    @staticmethod
    def valid_criterion(
        y_true: torch.Tensor, y_logits: torch.Tensor, criterion_params
    ):
        """Validation Loss Function.

        Args:
            y_true ([type]): [description]
            y_logits ([type]): [description]
            criterion_params (global_params.CriterionParams, optional): [description]. Defaults to CRITERION_PARAMS.criterion_params.

        Returns:
            [type]: [description]
        """
        loss_fn = getattr(torch.nn, criterion_params.valid_criterion_name)(
            **criterion_params.valid_criterion_params
        )
        loss = loss_fn(y_logits, y_true)
        return loss

    def get_classification_metrics(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
    ):
        """[summary]

        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);

        Returns:
            [type]: [description]
        """
        # TODO: To implement Ian's Results class here so that we can return as per the following link: https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        # TODO: To think whether include num_classes, threshold etc in the arguments.
        torchmetrics_accuracy = metrics.accuracy_score_torch(
            y_trues,
            y_preds,
            num_classes=self.params.num_classes,
            threshold=0.5,
        )

        auroc_dict = metrics.multiclass_roc_auc_score_torch(
            y_trues,
            y_probs,
            num_classes=self.params.num_classes,
        )

        _auroc_all_classes, macro_auc = (
            auroc_dict["auroc_per_class"],
            auroc_dict["macro_auc"],
        )

        # TODO: To check robustness of the code for confusion matrix.
        # macro_cm = metrics.tp_fp_tn_fn_binary(
        #     y_true=y_trues, y_prob=y_probs, class_labels=[0, 1, 2, 3, 4]
        # )

        return {"accuracy": torchmetrics_accuracy, "macro_auroc": macro_auc}

    @staticmethod
    def get_lr(optimizer: torch.optim) -> float:
        """Get the learning rate of the current epoch.
        Note learning rate can be different for different layers, hence the for loop.
        Args:
            self.optimizer (torch.optim): [description]
        Returns:
            float: [description]
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        fold: int = None,
    ):
        """Fit the model.

        Args:
            train_loader (torch.utils.data.DataLoader): [description]
            val_loader (torch.utils.data.DataLoader): [description]
            fold (int, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        # To automatically log gradients
        self.wandb_run.watch(self.model, log_freq=100)
        self.best_valid_loss = np.inf

        training_logger.info(
            f"\nTraining on Fold {fold} and using {self.params.model_name}\n"
        )

        for _epoch in range(1, self.params.epochs + 1):

            # get current epoch's learning rate
            curr_lr = self.get_lr(self.optimizer)

            ############################ Start of Training #############################

            train_start_time = time.time()

            train_one_epoch_dict = self.train_one_epoch(train_loader)
            train_loss = train_one_epoch_dict["train_loss"]

            # total time elapsed for this epoch
            train_time_elapsed = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - train_start_time)
            )

            training_logger.info(
                f"\n[RESULT]: Train. Epoch {_epoch}:\nAvg Train Summary Loss: {train_loss:.3f}\
                \nLearning Rate: {curr_lr:.5f}\nTime Elapsed: {train_time_elapsed}\n"
            )

            ########################### End of Training #################################

            ########################### Start of Validation #############################

            val_start_time = time.time()  # start time for validation
            valid_one_epoch_dict = self.valid_one_epoch(valid_loader)

            (
                valid_loss,
                valid_trues,
                valid_logits,
                valid_preds,
                valid_probs,
            ) = (
                valid_one_epoch_dict["valid_loss"],
                valid_one_epoch_dict["valid_trues"],
                valid_one_epoch_dict["valid_logits"],
                valid_one_epoch_dict["valid_preds"],
                valid_one_epoch_dict["valid_probs"],
            )

            # total time elapsed for this epoch
            valid_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - val_start_time)
            )

            valid_metrics_dict = self.get_classification_metrics(
                valid_trues,
                valid_preds,
                valid_probs,
            )

            valid_accuracy, valid_macro_auroc = (
                valid_metrics_dict["accuracy"],
                valid_metrics_dict["macro_auroc"],
            )

            # TODO: Still need save each metric for each epoch into a list history. Rename properly
            # TODO: Log each metric to wandb and log file.

            training_logger.info(
                f"\n[RESULT]: Validation. Epoch {_epoch}\nAvg Val Summary Loss: {valid_loss:.3f}\
                \nAvg Val Accuracy: {valid_accuracy:.3f}\nAvg Val Macro AUROC: {valid_macro_auroc:.3f}\
                \nTime Elapsed: {valid_elapsed_time}\n"
            )

            ########################### End of Validation ##############################

            ########################### Start of Wandb #################################
            self.history["epoch"].append(_epoch)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_accuracy"].append(valid_accuracy)
            self.history["valid_macro_auroc"].append(valid_macro_auroc)
            self.log_metrics(_epoch, self.history)
            ########################### End of Wandb ###################################

            ########################## Start of Early Stopping ##########################
            ########################## Start of Model Saving ############################
            # TODO: Consider condensing early stopping and model saving as callbacks, it looks very long and ugly here.

            # User has to choose a few metrics to monitor.
            # Here I chose valid_loss and valid_macro_auroc.
            self.monitored_metric["metric_score"] = torch.clone(
                valid_macro_auroc
            )
            if self.early_stopping is not None:
                # TODO: Implement this properly, Add save_model_artifacts here as well. Or rather, create a proper callback to reduce the complexity of code below.
                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=valid_loss
                )
                self.best_valid_loss = best_score

                if early_stop:
                    training_logger.info("Stopping Early!")
                    break
            else:

                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss

                if self.monitored_metric["mode"] == "max":
                    if (
                        self.monitored_metric["metric_score"]
                        > self.best_valid_score
                    ):
                        training_logger.info(
                            f"\nValidation {self.monitored_metric['metric_name']} improved from {self.best_valid_score} to {self.monitored_metric['metric_score']}"
                        )
                        self.best_valid_score = self.monitored_metric[
                            "metric_score"
                        ]
                        # Reset patience counter as we found a new best score
                        patience_counter_ = self.patience_counter
                        # TODO: Overwrite model saving whenever a better score is found. Currently this part is clumsy because we need to shift it to the else clause if we are monitoring min metrics. Do you think it is a good idea to put this chunk in save_model_artifacts instead?

                        saved_model_path = Path(
                            self.model_path,
                            f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
                        )
                        self.save_model_artifacts(
                            saved_model_path,
                            valid_trues,
                            valid_logits,
                            valid_preds,
                            valid_probs,
                        )
                        #  model_path = Path(wandb.run.dir, "model.pt").absolute().__str__()
                        # self.wandb_run.save(saved_model_path.__str__())
                        # TODO: Temporary workaround for Windows to save wandb files by copying to the local directory which will auto sync later. https://github.com/wandb/client/issues/1370
                        shutil.copy(
                            saved_model_path.__str__(),
                            os.path.join(
                                self.wandb_run.dir,
                                f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
                            ),
                        )

                        training_logger.info(
                            f"\nSaving model with best valid {self.monitored_metric['metric_name']} score: {self.best_valid_score}\n"
                        )
                    else:
                        patience_counter_ -= 1
                        training_logger.info(
                            f"Patience Counter {patience_counter_}"
                        )
                        if patience_counter_ == 0:
                            training_logger.info(
                                f"\n\nEarly Stopping, patience reached!\n\nbest valid {self.monitored_metric['metric_name']} score: {self.best_valid_score}"
                            )
                            break
                else:
                    if (
                        self.monitored_metric["metric_score"]
                        < self.best_valid_score
                    ):
                        self.best_valid_score = self.monitored_metric[
                            "metric_score"
                        ]

            ########################## End of Early Stopping ############################
            ########################## End of Model Saving ##############################

            ########################## Start of Scheduler ###############################

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

            ########################## End of Scheduler #################################

        ########################## Load Best Model ######################################
        # Load current checkpoint so we can get model's oof predictions, often in the form of probabilities.
        curr_fold_best_checkpoint = self.load(
            Path(
                self.model_path,
                f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
            )
        )
        ########################## End of Load Best Model ###############################

        ######################### Delete Optimizer and Scheduler ########################
        utils.free_gpu_memory(
            self.optimizer,
            self.scheduler,
            valid_trues,
            valid_logits,
            valid_preds,
            valid_probs,
        )
        ########################## End of Delete Optimizer and Scheduler ################

        return curr_fold_best_checkpoint

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Train one epoch of the model."""

        metric_monitor = metrics.MetricMonitor()

        # set to train mode
        self.model.train()
        average_cumulative_train_loss: float = 0.0
        train_bar = tqdm(train_loader)

        # Iterate over train batches
        for step, data in enumerate(train_bar, start=1):
            # TODO: Consider rename data to batch for names consistency.
            if self.params.mixup:
                # TODO: Implement MIXUP logic. Refer here: https://www.kaggle.com/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix and my https://colab.research.google.com/drive/1sYkKG8O17QFplGMGXTLwIrGKjrgxpRt5#scrollTo=5y4PfmGZubYp
                #       MIXUP logic can be found in petfinder.
                pass

            # unpack - note that if BCEWithLogitsLoss, dataset should do view(-1,1) and not here.
            inputs = data["X"].to(self.device, non_blocking=True)
            targets = data["y"].to(self.device, non_blocking=True)

            batch_size = inputs.shape[0]

            with torch.cuda.amp.autocast(
                enabled=self.params.use_amp,
                dtype=torch.float16,
                cache_enabled=True,
            ):
                logits = self.model(inputs)  # Forward pass logits
                curr_batch_train_loss = self.train_criterion(
                    targets,
                    logits,
                    batch_size,
                    criterion_params=self.pipeline_config.criterion_params,
                )
            self.optimizer.zero_grad()  # reset gradients

            if self.scaler is not None:
                self.scaler.scale(curr_batch_train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                curr_batch_train_loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights using the optimizer

            # Update loss metric
            metric_monitor.update("Loss", curr_batch_train_loss.item())
            train_bar.set_description(f"Train. {metric_monitor}")

            # Cumulative Loss
            # Batch/Step 1: curr_batch_train_loss = 10 -> average_cumulative_train_loss = (10-0)/1 = 10
            # Batch/Step 2: curr_batch_train_loss = 12 -> average_cumulative_train_loss = 10 + (12-10)/2 = 11 (Basically (10+12)/2=11)
            # Essentially, average_cumulative_train_loss = loss over all batches / batches
            average_cumulative_train_loss += (
                curr_batch_train_loss.detach().item()
                - average_cumulative_train_loss
            ) / (step)

            _y_train_prob = get_sigmoid_softmax(self.pipeline_config)(logits)
            _y_train_pred = torch.argmax(_y_train_prob, dim=1)

        # TODO: Consider enhancement that returns the same dict as valid_one_epoch.
        return {"train_loss": average_cumulative_train_loss}

    def valid_one_epoch(
        self, valid_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Validate the model on the validation set for one epoch.

        Args:
            valid_loader (torch.utils.data.DataLoader): The validation set dataloader.

        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set. shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set. shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set. shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set. shape = (num_samples, num_classes)
        """

        self.model.eval()  # set to eval mode
        metric_monitor = metrics.MetricMonitor()
        average_cumulative_valid_loss: float = 0.0
        valid_bar = tqdm(valid_loader)

        valid_logits, valid_trues, valid_preds, valid_probs = [], [], [], []

        with torch.no_grad():
            for step, data in enumerate(valid_bar, start=1):
                # unpack
                inputs = data["X"].to(self.device, non_blocking=True)
                targets = data["y"].to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                _batch_size = inputs.shape[0]

                # TODO: Refer to my RANZCR notes on difference between Softmax and Sigmoid with examples.
                y_valid_prob = get_sigmoid_softmax(self.pipeline_config)(logits)
                y_valid_pred = torch.argmax(y_valid_prob, axis=1)

                curr_batch_val_loss = self.valid_criterion(
                    targets,
                    logits,
                    criterion_params=self.pipeline_config.criterion_params,
                )

                metric_monitor.update(
                    "Loss", curr_batch_val_loss.item()
                )  # Update loss metric

                valid_bar.set_description(f"Validation. {metric_monitor}")
                average_cumulative_valid_loss += (
                    curr_batch_val_loss.item() - average_cumulative_valid_loss
                ) / (step)

                # For OOF score and other computation.
                # TODO: Consider giving numerical example. Consider rolling back to targets.cpu().numpy() if torch fails.
                valid_trues.extend(targets.cpu())
                valid_logits.extend(logits.cpu())
                valid_preds.extend(y_valid_pred.cpu())
                valid_probs.extend(y_valid_prob.cpu())

                # TODO: To make this look like what we did in breast cancer project.

        # We should work with numpy arrays.
        # vstack here to stack the list of numpy arrays.
        # a = [np.asarray([1,2,3]), np.asarray([4,5,6])]
        # np.vstack(a) -> array([[1, 2, 3], [4, 5, 6]])
        valid_trues, valid_logits, valid_preds, valid_probs = (
            torch.vstack(valid_trues),
            torch.vstack(valid_logits),
            torch.vstack(valid_preds),
            torch.vstack(valid_probs),
        )
        num_valid_samples = len(valid_trues)
        assert valid_trues.shape == valid_preds.shape == (num_valid_samples, 1)
        assert (
            valid_logits.shape
            == valid_probs.shape
            == (num_valid_samples, self.params.num_classes)
        )

        return {
            "valid_loss": average_cumulative_valid_loss,
            "valid_trues": valid_trues,
            "valid_logits": valid_logits,
            "valid_preds": valid_preds,
            "valid_probs": valid_probs,
        }

    def log_metrics(
        self, epoch: int, history: Dict[str, Union[float, np.ndarray]]
    ):
        """Log a scalar value to both MLflow and TensorBoard
        Args:
            history (Dict[str, Union[float, np.ndarray]]): A dictionary of metrics to log.
        """
        for metric_name, metric_values in history.items():
            self.wandb_run.log(
                {metric_name: metric_values[epoch - 1]}, step=epoch
            )

    def log_weights(self, step):
        """Log the weights of the model to both MLflow and TensorBoard.
        # TODO: Check https://github.com/ghnreigns/reighns-mnist/tree/master/reighns_mnist
        Args:
            step ([type]): [description]
        """
        self.writer.add_histogram(
            tag="conv1_weight",
            values=self.model.conv1.weight.data,
            global_step=step,
        )

    # TODO: Consider unpacking the dict returned by valid_one_epoch instead of passing in as arguments.
    def save_model_artifacts(
        self,
        path: str,
        valid_trues: torch.Tensor,
        valid_logits: torch.Tensor,
        valid_preds: torch.Tensor,
        valid_probs: torch.Tensor,
    ) -> None:
        """Save the weight for the best evaluation metric and also the OOF scores.

        Caution: I removed model.eval() here as this is not standard practice.

        valid_trues -> oof_trues: np.array of shape [num_samples, 1] and represent the true labels for each sample in current fold.
                                i.e. oof_trues.flattened()[i] = true label of sample i in current fold.
        valid_logits -> oof_logits: np.array of shape [num_samples, num_classes] and represent the logits for each sample in current fold.
                                i.e. oof_logits[i] = [logit_of_sample_i_in_current_fold_for_class_0, logit_of_sample_i_in_current_fold_for_class_1, ...]
        valid_preds -> oof_preds: np.array of shape [num_samples, 1] and represent the predicted labels for each sample in current fold.
                                i.e. oof_preds.flattened()[i] = predicted label of sample i in current fold.
        valid_probs -> oof_probs: np.array of shape [num_samples, num_classes] and represent the probabilities for each sample in current fold. i.e. first row is the probabilities of the first class.
                                i.e. oof_probs[i] = [probability_of_sample_i_in_current_fold_for_class_0, probability_of_sample_i_in_current_fold_for_class_1, ...]
        """

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "oof_trues": valid_trues,
                "oof_logits": valid_logits,
                "oof_preds": valid_preds,
                "oof_probs": valid_probs,
            },
            path,
        )

    @staticmethod
    def load(path: str):
        """Load a model checkpoint from the given path.
        Reason for using a static method: https://stackoverflow.com/questions/70052073/am-i-using-static-method-correctly/70052107#70052107
        """
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        return checkpoint
