from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import wandb

from config import config


@dataclass
class FilePaths:
    """Class to keep track of the files."""

    train_images: Path = Path(config.DATA_DIR, "train")
    test_images: Path = Path(config.DATA_DIR, "test")
    train_csv: Path = Path(config.DATA_DIR, "raw/train.csv")
    test_csv: Path = Path(config.DATA_DIR, "raw/test.csv")
    sub_csv: Path = Path(config.DATA_DIR, "raw/sample_submission.csv")
    folds_csv: Path = Path(config.DATA_DIR, "processed/train.csv")

    weight_path: Path = Path(config.MODEL_REGISTRY)
    wandb_dir: Path = Path(config.WANDB_DIR)
    global_params_path: Path = Path(config.CONFIG_DIR, "global_params.py")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_model_artifacts_path(self) -> Path:
        """Returns the model artifacts path.

        Returns:
            Path(model_artifacts_path) (Path): Model artifacts path.
        """
        # model_artifacts_path stores model weights, oof, etc. Note that now the model save path has wandb_run's group id appended for me to easily recover which run corresponds to which model.
        # create model directory if not exist and model_directory with run_id to identify easily.

        model_artifacts_path: Path = Path(
            self.weight_path,
            f"{ModelParams().model_name}_{WandbParams().group}",
        )
        Path.mkdir(model_artifacts_path, parents=True, exist_ok=True)
        # oof_csv: Path = Path(model_artifacts_path)
        return model_artifacts_path


@dataclass
class DataLoaderParams:
    """Class to keep track of the data loader parameters."""

    train_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": True,
            "collate_fn": None,
        }
    )
    valid_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

    test_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_len_train_loader(self) -> int:
        """Returns the length of the train loader.

        This is useful when using OneCycleLR.

        Returns:
            int(len_of_train_loader) (int): Length of the train loader.
        """
        total_rows = pd.read_csv(FilePaths().train_csv).shape[
            0
        ]  # get total number of rows/images
        total_rows_per_fold = total_rows / (MakeFolds().num_folds)
        total_rows_per_training = total_rows_per_fold * (
            MakeFolds().num_folds - 1
        )  # if got 1000 images, 10 folds, then train on 9 folds = 1000/10 * (10-1) = 100 * 9 = 900
        len_of_train_loader = (
            total_rows_per_training // self.train_loader["batch_size"]
        )  # if 900 rows, bs is 16, then 900/16 = 56.25, but we drop last if dataloader, so become 56 steps. if not 57 steps.
        return int(len_of_train_loader)


@dataclass
class MakeFolds:
    """A class to keep track of cross-validation schema.

    seed (int): random seed for reproducibility.
    num_folds (int): number of folds.
    cv_schema (str): cross-validation schema.
    class_col_name (str): name of the target column.
    image_col_name (str): name of the image column.
    folds_csv (str): path to the folds csv.
    """

    seed: int = 1992
    num_folds: int = 5
    cv_schema: str = "StratifiedGroupKFold"
    class_col_name: str = "target"
    image_col_name: str = "image_name"
    image_extension: str = ".jpg"
    group_kfold_split: str = "patient_id"
    folds_csv: Path = FilePaths().folds_csv

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AugmentationParams:
    """Class to keep track of the augmentation parameters."""

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    image_size: int = 256
    mixup: bool = False
    mixup_params: Dict[str, Any] = field(
        default_factory=lambda: {"mixup_alpha": 1, "use_cuda": True}
    )
    hairs_folder: Path = Path.joinpath(config.DATA_DIR, "melanoma_hairs")
    use_hair_aug: bool = True
    use_microscope_aug: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CriterionParams:
    """A class to track loss function parameters."""

    train_criterion_name: str = "CrossEntropyLoss"
    valid_criterion_name: str = "CrossEntropyLoss"
    train_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.0,
        }
    )
    valid_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.0,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelParams:
    """A class to track model parameters.

    model_name (str): name of the model.
    pretrained (bool): If True, use pretrained model.
    input_channels (int): RGB image - 3 channels or Grayscale 1 channel
    output_dimension (int): Final output neuron.
                      It is the number of classes in classification.
                      Caution: If you use sigmoid layer for Binary, then it is 1.
    classification_type (str): classification type.
    """

    model_name: str = "tf_efficientnet_b0_ns"  # resnet50d resnext50_32x4d "tf_efficientnet_b0_ns"  # Debug use tf_efficientnet_b0_ns else tf_efficientnet_b4_ns

    pretrained: bool = True
    input_channels: int = 3
    output_dimension: int = 2
    classification_type: str = "multiclass"
    use_meta: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def check_dimension(self) -> None:
        """Check if the output dimension is correct."""
        if (
            self.classification_type == "binary"
            and CriterionParams().train_criterion_name == "BCEWithLogitsLoss"
        ):
            assert self.output_dimension == 1, "Output dimension should be 1"
        elif self.classification_type == "multilabel":
            config.logger.info(
                "Check on output dimensions as we are likely using BCEWithLogitsLoss"
            )


@dataclass
class GlobalTrainParams:

    debug: bool = True
    debug_multiplier: int = 128
    epochs: int = 10  # 10 when not debug
    use_amp: bool = True
    mixup: bool = AugmentationParams().mixup
    patience: int = 3
    model_name: str = ModelParams().model_name
    num_classes: int = ModelParams().output_dimension
    classification_type: str = ModelParams().classification_type
    use_hair_aug: bool = AugmentationParams().use_hair_aug
    use_microscope_aug: bool = AugmentationParams().use_microscope_aug
    use_meta: bool = ModelParams().use_meta
    meta_features: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptimizerParams:
    """A class to track optimizer parameters.

    optimizer_name (str): name of the optimizer.
    lr (float): learning rate.
    weight_decay (float): weight decay.
    """

    # batch size increase 2, lr increases a factor of 2 as well.
    optimizer_name: str = "AdamW"
    optimizer_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "amsgrad": False,
            "weight_decay": 1e-6,
            "eps": 1e-08,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SchedulerParams:
    """A class to track Scheduler Params."""

    scheduler_name: str = "CosineAnnealingWarmRestarts"  # Debug
    # scheduler_name: str = "OneCycleLR"
    if scheduler_name == "CosineAnnealingWarmRestarts":

        scheduler_params: Dict[str, Any] = field(
            default_factory=lambda: {
                "T_0": 10,
                "T_mult": 1,
                "eta_min": 1e-6,
                "last_epoch": -1,
            }
        )
    elif scheduler_name == "OneCycleLR":
        scheduler_params: Dict[str, Any] = field(
            default_factory=lambda: {
                "max_lr": 3e-4,
                "steps_per_epoch": DataLoaderParams().get_len_train_loader(),
                "epochs": GlobalTrainParams().epochs,
                "last_epoch": -1,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class WandbParams:
    """A class to track wandb parameters."""

    project: str = "siim-isic-melanoma-classification"
    entity: str = "reighns"
    save_code: bool = True
    job_type: str = "Train"
    # add an unique group id behind group name.
    group: str = f"{GlobalTrainParams().model_name}_{MakeFolds().num_folds}_folds_{wandb.util.generate_id()}"
    dir: str = FilePaths().wandb_dir
    print(f"wandb run group: {group}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LogsParams:
    """A class to track logging parameters."""

    # TODO: Slightly unclear as we decouple the mkdir logic from config.py. May consider to move it to config.py somehow.
    # What is preventing this is I need to pass in the run id from WANDB to the logs folder. Same happens in trainer.py when creating model dir.
    LOGS_DIR_RUN_ID = Path.joinpath(
        config.LOGS_DIR, f"run_id_{WandbParams().group}"
    )
    Path.mkdir(LOGS_DIR_RUN_ID, parents=True, exist_ok=True)

    if not LOGS_DIR_RUN_ID.exists():
        Path.mkdir(LOGS_DIR_RUN_ID, parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PipelineConfig:

    files: FilePaths
    loader_params: DataLoaderParams
    folds: MakeFolds
    transforms: AugmentationParams
    criterion_params: CriterionParams
    model_params: ModelParams
    global_train_params: GlobalTrainParams
    optimizer_params: OptimizerParams
    scheduler_params: SchedulerParams
    wandb_params: WandbParams
    logs_params: LogsParams

    def __init__(
        self,
        files,
        loader_params,
        folds,
        transforms,
        criterion_params,
        model_params,
        global_train_params,
        optimizer_params,
        scheduler_params,
        wandb_params,
        logs_params,
    ):

        self.files = files
        self.loader_params = loader_params
        self.folds = folds
        self.transforms = transforms
        self.criterion_params = criterion_params
        self.model_params = model_params
        self.global_train_params = global_train_params
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.wandb_params = wandb_params
        self.logs_params = logs_params
