import functools
from pathlib import Path
from typing import Callable, Dict, OrderedDict, Tuple, Union

import timm
import torch
import torchsummary
from config import config, global_params

from src import utils

MODEL_PARAMS = global_params.ModelParams()
LOGS_PARAMS = global_params.LogsParams()
device = config.DEVICE

models_logger = config.init_logger(
    log_file=Path.joinpath(LOGS_PARAMS.LOGS_DIR_RUN_ID, "models.log"),
    module_name="models",
)

# TODO: To check with Ian the best way to put pipeline_config in. The issue is if I put pipeline_config in the constructor
# then it is not easily modifiable outside. It happens that many places I need to define the model as pretrained=False and
# having the pipeline object makes it hard to modify.


class CustomNeuralNet(torch.nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_PARAMS.model_name,
        out_features: int = MODEL_PARAMS.output_dimension,
        in_channels: int = MODEL_PARAMS.input_channels,
        pretrained: bool = MODEL_PARAMS.pretrained,
        use_meta: bool = MODEL_PARAMS.use_meta,
    ):
        """Construct a new model.

        Args:
            model_name ([type], str): The name of the model to use. Defaults to MODEL_PARAMS.model_name.
            out_features ([type], int): The number of output features, this is usually the number of classes, but if you use sigmoid, then the output is 1. Defaults to MODEL_PARAMS.output_dimension.
            in_channels ([type], int): The number of input channels; RGB = 3, Grayscale = 1. Defaults to MODEL_PARAMS.input_channels.
            pretrained ([type], bool): If True, use pretrained model. Defaults to MODEL_PARAMS.pretrained.
        """
        super().__init__()

        self.in_channels = in_channels
        self.pretrained = pretrained
        self.use_meta = use_meta

        self.backbone = timm.create_model(
            model_name, pretrained=self.pretrained, in_chans=self.in_channels
        )
        models_logger.info(
            f"\nModel: {model_name}\nPretrained: {pretrained}\nIn Channels: {in_channels}\n"
        )

        # removes head from backbone: # TODO: Global pool = "avg" vs "" behaves differently in shape, caution!
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

        # get the last layer's number of features in backbone (feature map)
        self.in_features = self.backbone.num_features
        self.out_features = out_features

        # Custom Head
        # self.single_head_fc = torch.nn.Sequential(
        #     torch.nn.Linear(self.in_features, self.in_features),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(self.in_features, self.out_features),
        # )
        self.single_head_fc = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features),
        )

        if self.use_meta:
            pass

        self.architecture: Dict[str, Callable] = {
            "backbone": self.backbone,
            "bottleneck": None,
            "head": self.single_head_fc,
        }

    def extract_features(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """Extract the features mapping logits from the model.
        This is the output from the backbone of a CNN.

        Args:
            image (torch.FloatTensor): The input image.

        Returns:
            feature_logits (torch.FloatTensor): The features logits.
        """
        # TODO: To rename feature_logits to image embeddings, also find out what is image embedding.
        feature_logits = self.architecture["backbone"](image)
        return feature_logits

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """The forward call of the model.

        Args:
            image (torch.FloatTensor): The input image.

        Returns:
            classifier_logits (torch.FloatTensor): The output logits of the classifier head.
        """

        feature_logits = self.extract_features(image)
        classifier_logits = self.architecture["head"](feature_logits)

        return classifier_logits


def torchsummary_wrapper(
    model: CustomNeuralNet, image_size: Tuple[int, int, int]
) -> torchsummary.model_statistics.ModelStatistics:
    """A torch wrapper to print out layers of a Model.

    Args:
        model (CustomNeuralNet): Model.
        image_size (Tuple[int, int, int]): Image size as a tuple of (channels, height, width).

    Returns:
        model_summary (torchsummary.model_statistics.ModelStatistics): Model summary.
    """

    model_summary = torchsummary.summary(model, image_size)
    return model_summary


def forward_pass(
    loader: torch.utils.data.DataLoader,
    model: CustomNeuralNet,
) -> Union[
    torch.FloatTensor,
    torch.LongTensor,
    torchsummary.model_statistics.ModelStatistics,
]:
    """Performs a forward pass of a tensor through the model.

    Args:
        loader (torch.utils.data.DataLoader): The dataloader.
        model (CustomNeuralNet): Model to be used for the forward pass.

    Returns:
        X (torch.FloatTensor): The input tensor.
        y (torch.LongTensor): The output tensor.
    """
    utils.seed_all()
    model.to(device)
    batch_size, channel, height, width = iter(loader).next()["X"].shape
    image_size = (channel, height, width)

    try:
        models_logger.info("Model Summary:")
        torchsummary.summary(model, image_size)
    except RuntimeError:
        models_logger.debug(f"The channel is {channel}. Check!")

    X = torch.randn((batch_size, *image_size)).to(device)
    y = model(image=X)
    models_logger.info("Forward Pass Successful!")
    models_logger.info(f"X: {X.shape} \ny: {y.shape}")
    models_logger.info(f"X[0][0][0]: {X[0][0][0][0]} \ny[0][0][0]: {y[0][0]}")

    utils.free_gpu_memory(model, X, y)
    return X, y
