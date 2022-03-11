from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def unnormalize(
    normalized_img, mean, std, max_pixel_value=255.0
) -> torch.Tensor:
    """TODO: Use https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/7 code and make it a class to include both Normalize and Unnormalize Method.

    Formula:
    Normalize: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
    Unnormalize: img = img * std * max_pixel_value + mean * max_pixel_value

    Args:
        normalized_img ([type]): [description]
        mean ([type]): [description]
        std ([type]): [description]
        max_pixel_value (float, optional): [description]. Defaults to 255.0.

    Returns:
        torch.Tensor: [description]
    """
    # normalized_img = (unnormalized_img - mean * max_pixel_value) / (std * max_pixel_value)
    # unnormalized_img = normalized_img * (std * max_pixel_values) + mean * max_pixel_values

    unnormalized = torch.zeros(normalized_img.size(), dtype=torch.float64)
    unnormalized[0, :, :] = (
        normalized_img[0, :, :] * (std[0] * max_pixel_value)
        + mean[0] * max_pixel_value
    )
    unnormalized[1, :, :] = (
        normalized_img[1, :, :] * (std[1] * max_pixel_value)
        + mean[1] * max_pixel_value
    )
    unnormalized[2, :, :] = (
        normalized_img[2, :, :] * (std[2] * max_pixel_value)
        + mean[2] * max_pixel_value
    )

    return unnormalized


def show_image(
    loader: torch.utils.data.DataLoader,
    mean: List[float] = None,
    std: List[float] = None,
    one_channel: bool = False,
):
    """Plot a grid of image from Dataloader.

    Mutable Default Arguments are not encouraged, but I won't be using operations like append inside the func.

    Args:
        loader (torch.utils.data.DataLoader): The dataloader to be used.
        mean (List[float], optional): Here we are using ImageNet mean. Defaults to [0.485, 0.456, 0.406].
        std (List[float], optional): Here we are using ImageNet std. Defaults to [0.229, 0.224, 0.225].
        one_channel (bool, optional): If True, treat as grayscale. Defaults to False.
    """

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    dataiter = iter(loader)

    one_batch_images, one_batch_targets = (
        dataiter.next()["X"],
        dataiter.next()["y"],
    )

    one_batch_images = [
        unnormalize(image, mean, std, max_pixel_value=255.0)
        for image in one_batch_images
    ]

    # create grid of images
    image_grid = torchvision.utils.make_grid(one_batch_images, normalize=False)

    if one_channel:
        pass

    # Necessary to cast to int if not it will not show properly.
    image_grid = image_grid.numpy().astype(int)
    plt.figure(figsize=(20, 10))

    if one_channel:
        plt.imshow(image_grid, cmap="Greys")
    else:
        plt.imshow(np.transpose(image_grid, (1, 2, 0)))

    # TODO: Consider add label and image id name beside title. https://discuss.pytorch.org/t/add-label-captions-to-make-grid/42863/4
    plt.title(f"Labels: {one_batch_targets.numpy()}")
    plt.show()

    return image_grid
