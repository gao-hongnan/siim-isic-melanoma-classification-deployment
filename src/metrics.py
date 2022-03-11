from collections import defaultdict
import torch
from sklearn import metrics
from typing import List, Union, Dict
from config import config, global_params
import numpy as np
from torch.autograd.grad_mode import F
import torchmetrics
from torchmetrics.classification import auroc
import pandas as pd
from pathlib import Path
import collections


FOLDS = global_params.MakeFolds()
TRAIN_PARAMS = global_params.GlobalTrainParams()
LOGS_PARAMS = global_params.LogsParams()

# TODO: Add regression metrics.

metrics_logger = config.init_logger(
    log_file=Path.joinpath(LOGS_PARAMS.LOGS_DIR_RUN_ID, "metrics.log"),
    module_name="metrics",
)


class AverageLossMeter:
    """
    Computes and stores the average and current loss
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """To check PyTorch's code for resetting the loss meter."""

        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        """To check PyTorch's code for updating the loss meter.

        Args:
            curr_batch_avg_loss (float): _description_
            batch_size (str): _description_
        """
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count


class MetricMonitor:
    """Monitor Metrics"""

    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """To check PyTorch's code for resetting the metric monitor."""
        self.metrics = defaultdict(
            lambda: {"metric_score": 0, "count": 0, "average_score": 0}
        )

    def update(self, metric_name, metric_score):
        """To check PyTorch's code for updating the loss meter.

        Args:
            metric_name (_type_): _description_
            metric_score (_type_): _description_
        """
        metric = self.metrics[metric_name]

        metric["metric_score"] += metric_score
        metric["count"] += 1
        metric["average_score"] = metric["metric_score"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["average_score"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def accuracy_score_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute accuracy score for classification.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.
        threshold (float): Threshold for classification, note this can be used in binary classification.
                           For multi-class classification, one may need to use the macro-average.

    Returns:
        accuracy_score (torch.Tensor): Accuracy score.
    """
    accuracy_score = torchmetrics.Accuracy(
        threshold=threshold,
        num_classes=num_classes,
        average="micro",
        top_k=None,
    )(y_pred, y_true)
    return accuracy_score


def multiclass_roc_auc_score_torch(
    y_true: torch.Tensor, y_prob: torch.Tensor, num_classes: int
) -> Union[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
    """Compute ROC-AUC score for each class in a multiclass dataset.
    Reference: My website on classification metrics, the idea below is the same, with some subtle conveniences,
    the torchmetrics.ROC() class returns the fpr, tpr, thresholds, but unlike scikit-learn's
    roc_curve(), the fpr, tpr, thresholds are in the shape of (num_samples, num_classes),
    where first row of fpr, tpr and thresholds are for the first class, and second row is for the second class, and so on.
    Therefore, I do not need to label_binarize the y_true to get the fpr and tpr of each class as they have done it internally.
    In this case, I simply loop over zip(fpr, tpr) and compute the auc score for each class.

    This supports macro average only.

    The result should also equals to sklearn.metrics.roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')

    Args:
        y_true (np.ndarray of shape (n_samples, n_classes)) True labels. Caution: y_true will be flattened to 1-D as torchmetrics.roc_auc_score expects 1-D labels.
        y_prob (np.ndarray of shape (n_samples, n_classes)) Target scores
        classes (array-like of shape (n_classes,)) List of dataset classes. If `None`,
            the lexicographical order of the labels in `y_true` is used.

    Returns:
        auroc_per_class (array-like): ROC-AUC score for each class, in the same order as `classes`
        macro_auc (torch.Tensor): macro-average ROC-AUC score
    """

    auroc_per_class = []
    roc = torchmetrics.ROC(num_classes=num_classes)

    # flatten to 1-D as torchmetrics.roc_auc_score expects 1-D labels.
    y_true = torch.flatten(y_true)

    fpr_all_classes, tpr_all_classes, _ = roc(y_prob, y_true)
    for fpr, tpr in zip(fpr_all_classes, tpr_all_classes):
        curr_class_auroc = torchmetrics.AUC(reorder=True)(fpr, tpr)
        auroc_per_class.append(curr_class_auroc)

    macro_auroc = torch.mean(torch.stack(auroc_per_class), dim=0)

    assert torch.isclose(
        macro_auroc,
        torchmetrics.AUROC(
            num_classes=TRAIN_PARAMS.num_classes, average="macro"
        )(y_prob, y_true),
        rtol=1e-05,
        atol=1e-08,
        equal_nan=False,
    ), "The macro average should equals to torchmetrics's roc_auc_score."

    return {"auroc_per_class": auroc_per_class, "macro_auc": macro_auroc}


def brier_loss_binary_torch():
    """[summary]"""


def multiclass_label_binarize_torch(
    y: torch.Tensor, class_labels: List[int], pos_label=1, neg_label=0
):
    """Binarize labels in one-vs-all fashion.
    See: https://ghnreigns.github.io/reighns-ml-website/metrics/classification_metrics/precision_recall_f1/#building-our-own-classification-report

    Args:
        y (np.ndarray) Sequence of integer labels to encode
        class_labels (array-like) Labels for each class
        pos_label (int) Value for positive labels
        neg_label (int) Value for negative labels
    Returns:
        np.ndarray of shape (n_samples, n_classes) Encoded dataset
    """
    if isinstance(y, list):
        y = torch.as_tensor(y)

    binarized_cols = []

    for label in class_labels:
        binarize_col = torch.where(y == label, pos_label, neg_label)
        binarized_cols.append(binarize_col)

    return torch.column_stack(binarized_cols)


def tp_fp_tn_fn_binary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor = None,
    y_prob: torch.Tensor = None,
    class_labels: List[int] = None,
    threshold: float = 0.5,
    reduce: str = "macro",
):
    """Compute true positive, false positive, true negative, false negative.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        tp (torch.Tensor): True positive.
        fp (torch.Tensor): False positive.
        tn (torch.Tensor): True negative.
        fn (torch.Tensor): False negative.
    """

    if y_pred is not None and y_pred.dtype not in [torch.int64, torch.int32]:
        # raise dtype error
        raise TypeError(
            f"{y_pred.dtype} should be one of torch.int64 or torch.int32."
        )

    if y_prob is not None and y_prob.dtype not in [
        torch.float32,
        torch.float64,
    ]:
        raise TypeError(
            f"{y_prob.dtype} should be one of torch.float32 or torch.float64."
        )

    if y_pred is None and y_prob is None:
        raise ValueError("Either y_pred or y_prob should be provided.")

    if y_pred is not None:
        metrics_logger.info("Using y_pred to calculate tp, fp, tn, fn.")
        assert y_prob is None, "y_prob should be None if y_pred is provided."
        macro_cm = torchmetrics.StatScores(reduce=reduce, num_classes=2)(
            y_pred, y_true
        )
    else:
        metrics_logger.info("Using y_prob to calculate tp, fp, tn, fn.")
        assert y_pred is None, "y_pred should be None if y_prob is provided."

        macro_cm_dict = {}
        y_true_binarized = multiclass_label_binarize_torch(
            y_true, class_labels, 1, 0
        )
        for label in class_labels:
            curr_class_y_true_binarized = y_true_binarized[:, label]
            curr_class_y_prob = y_prob[:, label]
            curr_class_y_pred_binarized = torch.where(
                curr_class_y_prob > threshold, 1, 0
            )

            curr_class_cm = torchmetrics.StatScores()(
                curr_class_y_pred_binarized, curr_class_y_true_binarized
            )
            macro_cm_dict[label] = curr_class_cm
        return macro_cm_dict
    return macro_cm


# TODO: TO SIMPLIFY!
def calculate_cv_metrics(df_oof):
    """Calculate cross validation metrics."""

    num_classes = TRAIN_PARAMS.num_classes
    num_folds = FOLDS.num_folds
    accuracy_per_fold, macro_roc_auc_per_fold = [], []

    for fold in range(1, num_folds + 1):
        metrics_logger.info(f"Calculating metrics for fold {fold}.")

        y_true, y_probs, y_preds = (
            df_oof.loc[df_oof["fold"] == fold, "oof_trues"].values,
            df_oof.loc[
                df_oof["fold"] == fold,
                [f"class_{str(c)}_oof" for c in range(num_classes)],
            ].values,
            df_oof.loc[df_oof["fold"] == fold, "oof_preds"].values,
        )

        y_trues, y_probs, y_preds = (
            torch.from_numpy(y_true),
            torch.from_numpy(y_probs),
            torch.from_numpy(y_preds),
        )

        # TODO: Find a way to make threshold work, as of now, multiclass won't work.
        acc = torchmetrics.Accuracy(
            num_classes=num_classes, threshold=0.5, average="micro"
        )(y_preds, y_trues)

        auroc_dict = multiclass_roc_auc_score_torch(
            y_trues,
            y_probs,
            num_classes=num_classes,
        )

        _auroc_all_classes, macro_auc = (
            auroc_dict["auroc_per_class"],
            auroc_dict["macro_auc"],
        )
        accuracy_per_fold.append(acc)
        macro_roc_auc_per_fold.append(macro_auc)

    cv_mean_acc = np.mean(accuracy_per_fold)
    cv_mean_auroc = np.mean(macro_roc_auc_per_fold)
    cv_std_acc = np.std(accuracy_per_fold)
    cv_std_auroc = np.std(macro_roc_auc_per_fold)

    cv_mean_d = collections.OrderedDict()
    cv_std_d = collections.OrderedDict()
    cv_mean_d["cv_mean_acc"] = cv_mean_acc
    cv_mean_d["cv_mean_auroc"] = cv_mean_auroc
    cv_std_d["cv_std_acc"] = cv_std_acc
    cv_std_d["cv_std_auroc"] = cv_std_auroc

    return cv_mean_d, cv_std_d
