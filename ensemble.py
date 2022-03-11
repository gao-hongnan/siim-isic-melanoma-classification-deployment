import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from sklearn.metrics import roc_auc_score

# 1. As of March 2022, I still manually rename the oofs and subs to the data/oof_and_subs folder, consider automating this
# 2. It is also important to take y_trues from oof df, because taking from raw data might not be in correct sequence, since we may shuffle the data.


# pylint: disable=missing-docstring


def stack_oofs(
    oof_dfs: List[pd.DataFrame], pred_column_names: List[str]
) -> np.ndarray:
    """Stack all oof predictions horziontally.

    Args:
        oof_dfs (List[pd.DataFrame]): The list of oof predictions in dataframes.
        pred_column_names (List[str]): The list of prediction column names.

    Returns:
        all_oof_preds (np.ndarray): The stacked oof predictions of shape (num_samples, num_oofs * num_pred_columns).

    Example:
        >>> oof_1 = pd.DataFrame([1,2,3], columns=['class_1_oof'])
        >>> oof_2 = pd.DataFrame([4,5,6], columns=['class_1_oof'])
        >>> all_oof_preds = stack_oofs([oof_1, oof_2], ['class_1_oof'])
        >>> all_oof_preds = np.array([1, 4], [2, 5], [3, 6])
    """
    num_oofs = len(oof_dfs)
    num_samples = len(oof_dfs[0])
    num_target_cols = len(pred_column_names)
    all_oof_preds = np.zeros((num_samples, num_oofs * num_target_cols))

    if num_target_cols == 1:
        for index, oof_df in enumerate(oof_dfs):
            all_oof_preds[:, index : index + 1] = oof_df[
                pred_column_names
            ].values

    elif num_target_cols > 1:
        # Used in RANZCR where there are 11 target columns
        for index, oof_df in enumerate(oof_dfs):
            all_oof_preds[
                :, index * num_target_cols : (index + 1) * num_target_cols
            ] = oof_df[pred_column_names].values

    return all_oof_preds


def macro_multilabel_auc(
    label, pred, num_target_cols: int = 1, multilabel: bool = False
):
    """Also works for binary AUC like Melanoma"""

    if not multilabel:
        return roc_auc_score(label, pred)
    else:
        aucs = []
        for i in range(num_target_cols):
            print(label[:, i])
            print()
            print(pred[:, i])
            print(roc_auc_score(label[:, i], pred[:, i]))
        aucs.append(roc_auc_score(label, pred))

        return np.mean(aucs)


def compute_best_oof(
    all_oof_preds: np.ndarray,
    y_trues: np.ndarray,
    num_oofs: int,
    performance_metric: Callable,
) -> Tuple[float, int]:
    """Compute the oof score of all models using a performance metric and return the best model index and score.

    Args:
        all_oof_preds (np.ndarray): The stacked oof predictions of shape (num_samples, num_oofs * num_pred_columns). Taken from stack_oofs.
        y_trues (np.ndarray): The true labels of shape (num_samples, num_target_cols).
        num_oofs (int): The number of oof predictions.
        performance_metric (Callable): The performance metric to use, this is a function.

    Returns:
        best_oof_metric_score (float): The best oof score.
        best_model_index (int): The index of the best model.
    """
    all_oof_scores = []
    for k in range(num_oofs):
        metric_score = performance_metric(
            y_trues,
            all_oof_preds[:, k],
            num_target_cols=1,
            multilabel=False,
        )
        all_oof_scores.append(metric_score)
        print(f"Model {k} has OOF AUC = {metric_score}")

    best_oof_metric_score, best_oof_index = np.max(all_oof_scores), np.argmax(
        all_oof_scores
    )
    return best_oof_metric_score, best_oof_index


def calculate_best_score_over_weight_interval(
    weight_interval: float,
    model_i_oof: np.ndarray,
    model_j_oof: np.ndarray,
    y_trues: np.ndarray,
    performance_metric: Callable,
    running_best_score: float,
    running_best_weight: float,
    patience: int,
) -> Tuple[float, float]:
    """Calculate the best score over a weight interval.

    Args:
        weight_interval (float): _description_
        model_i_oof (np.ndarray): _description_
        model_j_oof (np.ndarray): _description_
        y_trues (np.ndarray): _description_
        performance_metric (Callable): _description_
        running_best_score (float): _description_
        running_best_weight (float): _description_
        patience (int): _description_

    Returns:
        Tuple[float, float]: _description_
    """
    patience_counter = 0
    for weight in range(weight_interval):
        temp_weight = weight / weight_interval

        temp_ensemble_oof_preds = (
            temp_weight * model_j_oof + (1 - temp_weight) * model_i_oof
        )

        temp_ensemble_oof_score = performance_metric(
            y_trues,
            temp_ensemble_oof_preds,
            num_target_cols=1,
            multilabel=False,
        )

        # in the first loop, if any of the blending is more than best_oof_metric_score, we will assign it to running_best_score.

        if temp_ensemble_oof_score > running_best_score:
            running_best_score = temp_ensemble_oof_score
            running_best_weight = temp_weight

        else:
            patience_counter += 1
            if patience_counter > patience:
                break

    return running_best_score, running_best_weight


def get_blended_oof(
    initial_best_model_oof, best_oof_index_list, best_weights_list
):
    # can be used on both oof and subs

    curr_model_oof = initial_best_model_oof
    for index, _ in enumerate(best_oof_index_list[1:]):
        model_j_index = best_oof_index_list[index + 1]

        curr_model_oof = (1 - best_weights_list[index]) * curr_model_oof + (
            best_weights_list[index]
        ) * all_oof_preds[:, model_j_index].reshape(-1, 1)
    return curr_model_oof


def return_list_of_dataframes(
    path: str, is_oof: bool = True
) -> Tuple[List[pd.DataFrame], int]:
    """Return a list of dataframes from a directory of files.

    The boolean is_oof is used to determine whether the list of dataframes contains oof or subs.

    Args:
        path (str): The path to the directory containing the files.
        is_oof (bool, optional): Determine whether the list of dataframes contains oof or subs. Defaults to True.

    Returns:
        List[pd.DataFrame]: The list of dataframes for either oof or subs.
        int: The number of files in the directory.
    """
    oof_and_subs_files = os.listdir(path)

    if is_oof:
        oof_files_sorted = np.sort(
            [f for f in oof_and_subs_files if "oof" in f]
        )
        return [
            pd.read_csv(os.path.join(path, k)) for k in oof_files_sorted
        ], len(oof_files_sorted)

    else:
        sub_files_sorted = np.sort(
            [f for f in oof_and_subs_files if "sub" in f]
        )
        return [
            pd.read_csv(os.path.join(path, k)) for k in sub_files_sorted
        ], len(sub_files_sorted)


if __name__ == "__main__":
    oof_and_subs_path = "./data/oof_and_subs"

    # ["oof_1.csv", "sub_1.csv", "oof_2.csv", "sub_2.csv"]
    oof_and_subs_files = os.listdir(oof_and_subs_path)

    # ["oof_1.csv", "oof_2.csv", "sub_1.csv", "sub_2.csv"] sorted
    oof_files_sorted = np.sort([f for f in oof_and_subs_files if "oof" in f])

    # [oof_1_df, oof_2_df, sub_1_df, sub_2_df] in dataframe
    oof_dfs_list = [
        pd.read_csv(os.path.join(oof_and_subs_path, k))
        for k in oof_files_sorted
    ]
    num_oofs = len(oof_dfs_list)

    # in my oof files, I also saved the corresponding y_trues, we thus take the first oof_df and use it to get the y_trues, assuming they are the same for all oof files.
    # note of caution, if you use different resampling methods, you will need to change the y_trues accordingly.

    y_trues_df = oof_dfs_list[0][["oof_trues"]]
    y_trues = y_trues_df.values
    print(f"We have {len(oof_files_sorted)} oof files.\n")

    target_cols = ["oof_trues"]
    pred_cols = ["class_1_oof"]
    num_target_cols = len(target_cols)
    all_oof_preds = stack_oofs(
        oof_dfs=oof_dfs_list, pred_column_names=pred_cols
    )
    print(
        f"all_oof_preds shape: {all_oof_preds.shape}\nThis variable is global and holds all oof predictions stacked horizontally.\n"
    )

    best_oof_metric_score, best_oof_index = compute_best_oof(
        all_oof_preds=all_oof_preds,
        y_trues=y_trues,
        num_oofs=num_oofs,
        performance_metric=macro_multilabel_auc,
    )

    print(
        f"\n### Computing Best OOF scores among all models ###\nThe best OOF AUC score is {best_oof_metric_score} and the best model index is {best_oof_index} corresponding to the oof file {oof_files_sorted[best_oof_index]}"
    )

    weight_interval = 1000  # 200
    patience = 20  # 10
    min_increase = 0.0003  # 0.00003

    print(
        f"\n### HyperParameters ###\nweight_interval = {weight_interval}\npatience = {patience}\nmin_increase = {min_increase}\n"
    )

    # keep track of oof index that are blended
    best_oof_index_list = [best_oof_index]
    best_weights_list = []

    print(f"Current Tracked Model List: {best_oof_index_list}")
    print(f"Current Weights List: {best_weights_list}")

    counter = 0

    # Initially, this curr_model_oof is the single best model that we got above from [oof_1, oof_2,...]
    initial_best_model_oof = all_oof_preds[:, best_oof_index].reshape(-1, 1)
    old_best_score = best_oof_metric_score
    model_i_best_score, model_i_index, model_i_weights = 0, 0, 0

    print(
        "Denote model i as the current model, and model j as the model that we are blending with."
    )
    for outer_oof_index in range(num_oofs):

        # basically in the first loop, we already know the current model's oof and we assign it by subsetting the all_oof_preds with the best oof index.
        curr_model_oof = initial_best_model_oof

        if counter > 0:
            curr_model_oof = get_blended_oof(
                initial_best_model_oof, best_oof_index_list, best_weights_list
            )

            print(curr_model_oof)

        for inner_oof_index in range(num_oofs):
            # If we have [oof_1, oof_2] and best_oof_index = 1 (oof_2), then we do not need to blend oof_2 and itself.

            if inner_oof_index in best_oof_index_list:
                continue
            # in the first loop, our running_best_score is the best_oof_metric_score
            # also our old_best_score is the best_oof_metric_score in the first loop
            (running_best_score, running_best_weight, patience_counter,) = (
                0,
                0,
                0,
            )
            # what we are doing here is to find the best oof score among all models that we have not blended yet.
            # for example, if we have [oof_1, oof_2, oof_3], and we know oof_2 is our initial_best_model_oof,
            # then we need to blend oof_2 with oof_1, then oof_2 with oof_3 to find out which of them yields the best overall oof when blended.
            (
                running_best_score,
                running_best_weight,
            ) = calculate_best_score_over_weight_interval(
                weight_interval,
                curr_model_oof,
                all_oof_preds[:, inner_oof_index].reshape(-1, 1),
                y_trues,
                macro_multilabel_auc,
                running_best_score,
                running_best_weight,
                patience,
            )

            if running_best_score > model_i_best_score:
                model_i_index = inner_oof_index
                model_i_best_score = running_best_score
                model_i_weights = running_best_weight

        increment = model_i_best_score - old_best_score
        if increment <= min_increase:
            print("Increment is too small, stop blending")
            break

        # DISPLAY RESULTS
        print()
        print(
            "Ensemble AUC = %.4f after adding model %i with weight %.3f. Increase of %.4f"
            % (
                model_i_best_score,
                model_i_index,
                model_i_weights,
                increment,
            )
        )
        print()

        old_best_score = model_i_best_score
        best_oof_index_list.append(model_i_index)

        best_weights_list.append(model_i_weights)
        print(f"Current Tracked Model List: {best_oof_index_list}")
        print(f"Current Weights List: {best_weights_list}")
        print(f"Current Best Score: {model_i_best_score}")
        counter += 1

    plt.hist(curr_model_oof, bins=100)
    plt.title("Ensemble OOF predictions")
    plt.show()

    # apply on submission
    sub_files_sorted = np.sort([f for f in oof_and_subs_files if "sub" in f])
    sub_dfs_list = [
        pd.read_csv(os.path.join(oof_and_subs_path, k))
        for k in sub_files_sorted
    ]

    print(f"\nWe have {len(sub_files_sorted)} submission files...")
    print()
    print(sub_files_sorted)

    y = np.zeros((len(sub_dfs_list[0]), len(sub_files_sorted) * len(pred_cols)))
    print(y.shape)
    for k in range(len(sub_files_sorted)):
        y[
            :, int(k * len(pred_cols)) : int((k + 1) * len(pred_cols))
        ] = sub_dfs_list[k]["target"].values.reshape(-1, 1)

    print(y)
    md2 = y[
        :,
        int(best_oof_index_list[0] * len(pred_cols)) : int(
            (best_oof_index_list[0] + 1) * len(pred_cols)
        ),
    ]
    print(md2)
    for i, k in enumerate(best_oof_index_list[1:]):
        md2 = (
            best_weights_list[i]
            * y[:, int(k * len(pred_cols)) : int((k + 1) * len(pred_cols))]
            + (1 - best_weights_list[i]) * md2
        )
    plt.hist(md2, bins=100)
    plt.show()
    df = sub_dfs_list[0].copy()
    df[["target"]] = md2
    df.to_csv("submission.csv", index=False)
    df.head()
