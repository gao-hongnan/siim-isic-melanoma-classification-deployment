import os
import random
from typing import Dict, Union

import albumentations
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from config import global_params
from albumentations.core.transforms_interface import ImageOnlyTransform

# pylint: disable=W0223


class AdvancedHairAugmentation(ImageOnlyTransform):
    """
    Impose an image of a hair to the target image.

    https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(
        self, hairs: int = 4, hairs_folder: str = "", always_apply=False, p=0.5
    ):
        super(AdvancedHairAugmentation, self).__init__(
            always_apply=always_apply, p=p
        )
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def apply(self, img, **params):
        """_summary_

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        _height, _width, _ = img.shape  # target image width and height
        hair_images = [
            im for im in os.listdir(self.hairs_folder) if "png" in im
        ]

        for _ in range(n_hairs):
            hair = cv2.imread(
                os.path.join(self.hairs_folder, random.choice(hair_images))
            )
            hair = cv2.cvtColor(hair, cv2.COLOR_BGR2RGB)
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            _ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst
        return img

    def get_params_dependent_on_targets(self, params):
        """_summary_

        Args:
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ()


class Microscope(ImageOnlyTransform):
    """https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159476

    Args:
        ImageOnlyTransform (_type_): _description_
    """

    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        """_summary_

        Args:
            img (_type_): _description_

        Returns:
            _type_: _description_
        """
        if random.random() < self.p:
            circle = cv2.circle(
                (np.ones(img.shape) * 255).astype(np.uint8),
                (img.shape[0] // 2, img.shape[1] // 2),
                random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),
                (0, 0, 0),
                -1,
            )

            mask = circle - 255
            img = np.multiply(img, mask)

        return img


def get_train_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on training data.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        albumentations.core.composition.Compose: The transforms for training set.
    """
    if (
        pipeline_config.transforms.use_hair_aug
        and pipeline_config.transforms.use_microscope_aug
    ):
        return albumentations.Compose(
            [
                AdvancedHairAugmentation(
                    hairs_folder=pipeline_config.transforms.hairs_folder
                ),
                albumentations.RandomResizedCrop(
                    height=pipeline_config.transforms.image_size,
                    width=pipeline_config.transforms.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    p=1.0,
                ),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Cutout(
                    max_h_size=int(
                        pipeline_config.transforms.image_size * 0.375
                    ),
                    max_w_size=int(
                        pipeline_config.transforms.image_size * 0.375
                    ),
                    num_holes=1,
                    p=0.3,
                ),
                Microscope(p=0.5),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )
    else:
        return albumentations.Compose(
            [
                albumentations.RandomResizedCrop(
                    height=pipeline_config.transforms.image_size,
                    width=pipeline_config.transforms.image_size,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    p=1.0,
                ),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightness(limit=0.2, p=0.75),
                albumentations.RandomContrast(limit=0.2, p=0.75),
                albumentations.OneOf(
                    [
                        albumentations.MotionBlur(blur_limit=5),
                        albumentations.MedianBlur(blur_limit=5),
                        albumentations.GaussianBlur(blur_limit=5),
                        albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                    ],
                    p=0.7,
                ),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(distort_limit=1.0),
                        albumentations.GridDistortion(
                            num_steps=5, distort_limit=1.0
                        ),
                        albumentations.ElasticTransform(alpha=3),
                    ],
                    p=0.7,
                ),
                albumentations.CLAHE(clip_limit=4.0, p=0.7),
                albumentations.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5,
                ),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.85,
                ),
                albumentations.Cutout(
                    max_h_size=int(
                        pipeline_config.transforms.image_size * 0.375
                    ),
                    max_w_size=int(
                        pipeline_config.transforms.image_size * 0.375
                    ),
                    num_holes=1,
                    p=0.5,
                ),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )


def get_valid_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on validation data.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        albumentations.core.composition.Compose: The transforms for validation set.
    """
    return albumentations.Compose(
        [
            albumentations.Resize(
                pipeline_config.transforms.image_size,
                pipeline_config.transforms.image_size,
            ),
            albumentations.Normalize(
                mean=pipeline_config.transforms.mean,
                std=pipeline_config.transforms.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_gradcam_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on gradcam data.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        albumentations.core.composition.Compose: The transforms for gradcam.
    """
    return albumentations.Compose(
        [
            albumentations.Resize(
                pipeline_config.transforms.image_size,
                pipeline_config.transforms.image_size,
            ),
            albumentations.Normalize(
                mean=pipeline_config.transforms.mean,
                std=pipeline_config.transforms.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_inference_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> Dict[str, albumentations.core.composition.Compose]:
    """Performs Augmentation on test dataset.

    Remember tta transforms need resize and normalize.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        transforms_dict (Dict[str, albumentations.core.composition.Compose]): Returns the transforms for inference in a dictionary which can hold TTA transforms.
    """

    transforms_dict = {
        "transforms_test": albumentations.Compose(
            [
                albumentations.Resize(
                    pipeline_config.transforms.image_size,
                    pipeline_config.transforms.image_size,
                ),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        ),
        "tta_hflip": albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=1.0),
                albumentations.Resize(
                    pipeline_config.transforms.image_size,
                    pipeline_config.transforms.image_size,
                ),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        ),
        "tta_vflip": albumentations.Compose(
            [
                albumentations.VerticalFlip(p=1.0),
                albumentations.Resize(
                    pipeline_config.transforms.image_size,
                    pipeline_config.transforms.image_size,
                ),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        ),
        "microscope": albumentations.Compose(
            [
                Microscope(p=1),
                albumentations.Resize(
                    pipeline_config.transforms.image_size,
                    pipeline_config.transforms.image_size,
                ),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        ),
        "hair": albumentations.Compose(
            [
                AdvancedHairAugmentation(
                    hairs_folder=pipeline_config.transforms.hairs_folder, p=1
                ),
                albumentations.Resize(
                    pipeline_config.transforms.image_size,
                    pipeline_config.transforms.image_size,
                ),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        ),
    }

    return transforms_dict


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    pipeline_config: global_params.PipelineConfig,
) -> torch.Tensor:
    """Implements mixup data augmentation.

    Args:
        x (torch.Tensor): The input tensor.
        y (torch.Tensor): The target tensor.
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        mixup_params (TRANSFORMS, optional): [description]. Defaults to TRANSFORMS.mixup_params.

    Returns:
        torch.Tensor: [description]
    """

    mixup_params = pipeline_config.transforms.mixup_params

    # TODO: https://www.kaggle.com/reighns/petfinder-image-tabular check this to add z if there are dense targets.
    assert (
        mixup_params["mixup_alpha"] > 0
    ), "Mixup alpha must be greater than 0."
    assert (
        x.size(0) > 1
    ), "Mixup requires more than one sample as at least two samples are needed to mix."

    if mixup_params["mixup_alpha"] > 0:
        lambda_ = np.random.beta(
            mixup_params["mixup_alpha"], mixup_params["mixup_alpha"]
        )
    else:
        lambda_ = 1

    batch_size = x.size()[0]
    if mixup_params["use_cuda"] and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lambda_


def mixup_criterion(
    criterion: Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss],
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Implements mixup criterion.

    Args:
        criterion (Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss]): The loss function.
        logits (torch.Tensor): [description]
        y_a (torch.Tensor): [description]
        y_b (torch.Tensor): [description]
        lambda_ (float): [description]

    Returns:
        torch.Tensor: [description]
    """
    return lambda_ * criterion(logits, y_a) + (1 - lambda_) * criterion(
        logits, y_b
    )
