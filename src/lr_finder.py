from typing import Dict, List

import torch
from config import global_params
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter

from src import models, utils

CRITERION_PARAMS = global_params.CriterionParams()
OPTIMIZER_PARAMS = global_params.OptimizerParams()


# TODO: Find out how to use lr_finder when using Mixed Precision.


class CustomTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        return batch_data["X"], batch_data["y"]


class CustomValIter(ValDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        return batch_data["X"], batch_data["y"]


def find_lr(
    model: models.CustomNeuralNet,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    use_valid: bool = False,
) -> Dict[str, List[float]]:
    """[summary]

    Args:
        model (models.CustomNeuralNet): [description]
        device (torch.device): [description]
        train_loader (torch.utils.data.DataLoader): [description]
        valid_loader (torch.utils.data.DataLoader): [description]
        use_valid (bool, optional): [description]. Defaults to False.

    Returns:
        Dict[str, List[float]]: [description]
    """
    optimizer = getattr(torch.optim, OPTIMIZER_PARAMS.optimizer_name)(
        model.parameters(), **OPTIMIZER_PARAMS.optimizer_params
    )
    criterion = getattr(torch.nn, CRITERION_PARAMS.train_criterion_name)(
        **CRITERION_PARAMS.train_criterion_params
    )
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    custom_train_iter = CustomTrainIter(train_loader)
    lr_finder.reset()
    if use_valid:
        custom_valid_iter = CustomValIter(valid_loader)
        lr_finder.range_test(
            custom_train_iter,
            val_loader=custom_valid_iter,
            start_lr=1e-7,
            end_lr=3e-2,
            num_iter=100,
            step_mode="exp",
        )  # ["exp", "linear"]
    else:
        lr_finder.range_test(
            custom_train_iter,
            start_lr=1e-7,
            end_lr=3e-2,
            num_iter=100,
            step_mode="exp",
        )  # ["exp", "linear"]

    lr_finder.plot()

    utils.free_gpu_memory(optimizer, criterion, custom_train_iter)

    return lr_finder.history
