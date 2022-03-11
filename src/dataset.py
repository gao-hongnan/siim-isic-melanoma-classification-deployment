from typing import Dict, Union

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
from config import global_params


class CustomDataset(torch.utils.data.Dataset):
    """Dataset class for the {insert competition/project name} dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        pipeline_config: global_params.PipelineConfig,
        transforms: albumentations.core.composition.Compose = None,
        mode: str = "train",
    ):
        """Constructor for the dataset class.

        Args:
            df (pd.DataFrame): Dataframe for either train, valid or test.
            transforms (albumentations.core.composition.Compose): albumentations transforms to apply to the images.
            mode (str, optional): Defaults to "train". One of ['train', 'valid', 'test', 'gradcam']
        """

        self.pipeline_config = pipeline_config
        # "image_path" is hardcoded, as that is always defined in prepare_data.
        self.image_path = df["image_path"].values
        self.image_ids = df[self.pipeline_config.folds.image_col_name].values
        self.df = df
        self.targets = (
            torch.from_numpy(
                df[self.pipeline_config.folds.class_col_name].values
            )
            if mode != "test"
            else None
        )

        self.transforms = transforms
        self.mode = mode

        # this is None by default unless use_meta is True
        self.meta_features = (
            self.pipeline_config.global_train_params.meta_features
        )

        if self.mode not in ["train", "valid", "test", "gradcam"]:
            raise ValueError(
                f"Mode {self.mode} not in accepted list of modes {['train', 'valid', 'test', 'gradcam']}"
            )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def return_dtype(
        self, X: torch.Tensor, y: torch.Tensor, original_image: torch.Tensor
    ) -> torch.Tensor:
        """Return the dtype of the dataset.

        Args:
            X (torch.Tensor): Image tensor.
            y (torch.Tensor): Target tensor.
            original_image  (torch.Tensor): Original image tensor.

        Returns:
            X (torch.Tensor): Image tensor.
            y (torch.Tensor): Target tensor.
            original_image  (torch.Tensor): Original image tensor.
        """

        # Changing the shape of y tensor here is problematic, the dataloader somehow will turn a shape of [batch_size, 1] into [batch_size, 1, 1]
        # By definition of collate function in DataLoader, it is mentioned in documentation that it prepends an extra dimension to the tensor as batch_size. Thus,
        # if the input y is a tensor of shape [1,], then .view(-1, 1) will change the shape to [1, 1], and when we collate using DataLoader, say with batch_size = 4,
        # then the collated y will be a tensor of shape [4, 1, 1] instead of [4, 1] since it prepends an extra dimension.
        # TODO: Check on RANZCR to see if flatten here works since that is multi-label.

        if (
            self.pipeline_config.criterion_params.train_criterion_name
            == "BCEWithLogitsLoss"
        ):
            # Make changes to reshape rather than in Trainer.
            y = torch.as_tensor(y, dtype=torch.float32).flatten()
        else:
            y = torch.as_tensor(y, dtype=torch.long)

        X = torch.as_tensor(X, dtype=torch.float32)
        original_image = torch.as_tensor(original_image, dtype=torch.float32)

        return X, y, original_image

    def check_shape(self):
        """Check the shape of the dataset.

        Add a tensor transpose if transformation is None since most images is HWC but ToTensorV2 transforms them to CHW."""

        raise NotImplementedError

    def __getitem__(
        self, index: int
    ) -> Union[
        Dict[str, torch.FloatTensor],
        Dict[str, Union[torch.FloatTensor, torch.LongTensor]],
    ]:
        """Implements the getitem method: https://www.geeksforgeeks.org/__getitem__-and-__setitem__-in-python/

        Be careful of Targets:
            BCEWithLogitsLoss expects a target.float()
            CrossEntropyLoss expects a target.long()

        Args:
            index (int): index of the dataset.

        Returns:
            Dict[str, torch.FloatTensor]:{"X": image_tensor}
            Dict[str, Union[torch.FloatTensor, torch.LongTensor]]: {"y": target_tensor} If BCEwithLogitsLoss then FloatTensor, else LongTensor
        """
        image_path = self.image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # needed for gradcam.
        original_image = cv2.resize(
            image,
            (
                self.pipeline_config.transforms.image_size,
                self.pipeline_config.transforms.image_size,
            ),
        ).copy()

        # Get target for all modes except for test, if test, replace target with dummy ones to pass through return_dtype.
        target = self.targets[index] if self.mode != "test" else torch.ones(1)

        if self.transforms and isinstance(
            self.transforms, albumentations.core.composition.Compose
        ):
            image = self.transforms(image=image)["image"]

        # TODO: Consider not returning original image if we don't need it, may cause more memory usage and speed issues?
        X, y, original_image = self.return_dtype(image, target, original_image)

        if self.pipeline_config.global_train_params.use_meta:
            meta_features = np.asarray(
                self.df.iloc[index][self.meta_features].values, dtype=np.float32
            )
        else:
            # Cannot assign None as collate function will try to convert it to a tensor.
            meta_features = torch.ones(1)

        if self.mode in ["train", "valid"]:
            return {"X": X, "y": y, "meta_features": meta_features}

        if self.mode == "test":
            return {"X": X, "meta_features": meta_features}

        if self.mode == "gradcam":
            return {
                "X": X,
                "y": y,
                "original_image": original_image,
                "image_id": self.image_ids[index],
            }
