from enum import Enum
from typing import Union

import numpy as np

from pathlib import Path
from argparse import Namespace


class Mode(Enum):
    MIN = np.inf
    MAX = -np.inf


class EarlyStopping:
    """Class for Early Stopping."""

    mode_dict = {"min": np.inf, "max": -np.inf}

    def __init__(
        self, patience: int = 5, mode: Mode = Mode.MIN, min_delta: float = 1e-5
    ):
        """Construct an EarlyStopping instance.
        Arguments:
            patience : Number of epochs with no improvement after
                       which training will be stopped. (Default = 5)
            mode : One of {"min", "max"}. In min mode, training will
                   stop when the quantity monitored has stopped
                   decreasing.  In "max" mode it will stop when the
                   quantity monitored has stopped increasing.
            min_delta : Minimum change in the monitored quantity to
                        qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.stopping_counter = 0
        self.early_stop = False
        self.best_score = mode.value

    def improvement(
        self,
        curr_epoch_score: Union[float, int],
        curr_best_score: Union[float, int],
    ):
        if self.mode == Mode.MIN:
            return curr_epoch_score <= (curr_best_score - self.min_delta)
        return curr_epoch_score >= (curr_best_score + self.min_delta)

    @property
    def monitor_op(self):
        return self.mode.value

    def should_stop(self, curr_epoch_score):
        """The actual algorithm of early stopping.
        Arguments:
            epoch_score : The value of metric or loss which you montoring for that epoch.
            mode : The model which is being trained.
            model_path : The path to save the model.
            rmb false or true --> true, one is true is enough in boolean logic in or clause.
        """

        if self.improvement(
            curr_epoch_score=curr_epoch_score, curr_best_score=self.best_score
        ):

            # update self.best_score
            self.best_score = curr_epoch_score
            # reset patience once we have a new best score
            self.patience = params.patience

        else:
            self.stopping_counter += 1
            print(
                "Early Stopping Counter {} out of {}".format(
                    self.stopping_counter, self.patience
                )
            )

        if self.stopping_counter >= self.patience:

            print(
                "Early Stopping and since it is early stopping, we will not "
                "save the model since the metric has not improved for {} "
                "epochs".format(self.patience)
            )
            """self.early_stop flag is set to true, this will guarantee in the
            Trainer script, the training will break out of the loop"""
            self.early_stop = True

        return self.best_score, self.early_stop
