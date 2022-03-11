##### PREPARATIONS
from __future__ import generators, print_function

# libraries
import gc
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Union, OrderedDict

import albumentations
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch._C import device
from albumentations.pytorch.transforms import ToTensorV2
from config import config, global_params
from src import dataset, inference, models, prepare, utils

# TODO: 1. Now requirements.txt does not accept cuda version. https://discuss.streamlit.io/t/cant-find-dependencies-torch-1-9-0-cu102/14182

# download with progress bar
mybar = None
device = "cpu"


# Define global parameters to pass in PipelineConfig
FILES = global_params.FilePaths()
LOADER_PARAMS = global_params.DataLoaderParams()
FOLDS = global_params.MakeFolds()
TRANSFORMS = global_params.AugmentationParams()
MODEL_PARAMS = global_params.ModelParams()
GLOBAL_TRAIN_PARAMS = global_params.GlobalTrainParams()
WANDB_PARAMS = None
LOGS_PARAMS = global_params.LogsParams()
CRITERION_PARAMS = global_params.CriterionParams()
SCHEDULER_PARAMS = global_params.SchedulerParams()
OPTIMIZER_PARAMS = global_params.OptimizerParams()
INFERENCE_TRANSFORMS = global_params.AugmentationParams(image_size=256)

utils.seed_all(FOLDS.seed)

# INFERENCE_MODEL_PARAMS = global_params.ModelParams()
inference_pipeline_config = global_params.PipelineConfig(
    files=FILES,
    loader_params=LOADER_PARAMS,
    folds=FOLDS,
    transforms=INFERENCE_TRANSFORMS,
    model_params=MODEL_PARAMS,
    global_train_params=GLOBAL_TRAIN_PARAMS,
    wandb_params=WANDB_PARAMS,
    logs_params=LOGS_PARAMS,
    criterion_params=CRITERION_PARAMS,
    scheduler_params=SCHEDULER_PARAMS,
    optimizer_params=OPTIMIZER_PARAMS,
)


def show_progress(block_num, block_size, total_size):
    """Referenced from kozodoi's github:
    https://github.com/kozodoi/Pet_Pawpularity/blob/main/web_app.py

    Args:
        block_num (_type_): _description_
        block_size (_type_): _description_
        total_size (_type_): _description_
    """
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)


def streamlit_show_gradcam(
    model,
    image_tensor,
    original_image,
    state_dicts: List[OrderedDict],
    pipeline_config: global_params.PipelineConfig,
):
    """Log gradcam images into wandb for error analysis.
    # TODO: Consider getting the logits for error analysis, for example, if a predicted image which is correct has high logits this means the model is very sure, conversely, if a predicted image has low logits and also wrong, we also check why.
    """

    model_state_dict = state_dicts[0]  # we just want one state dict!

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    if "vit" in pipeline_config.model_params.model_name:
        # blocks[-1].norm1  # for vit models use this, note this is using TIMM backbone.
        target_layers = [model.backbone.blocks[-1].norm1]

    elif "efficientnet" in pipeline_config.model_params.model_name:
        target_layers = [model.backbone.conv_head]
        reshape_transform = None

    elif (
        "resnet" in pipeline_config.model_params.model_name
        or "resnext" in pipeline_config.model_params.model_name
    ):
        target_layers = [model.backbone.layer4[-1]]
        reshape_transform = None

    elif "swin" in pipeline_config.model_params.model_name:
        # https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/swinT_example.py
        # TODO: Note this does not work for swin 384 as the size is not (7, 7)
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(
                tensor.size(0), height, width, tensor.size(2)
            )

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.permute(0, 3, 1, 2)
            return result

        target_layers = [model.backbone.layers[-1].blocks[-1].norm1]

    # if image tensor is 3 dim, unsqueeze it to 4 dim with 1 in front.
    image_tensor = image_tensor.unsqueeze(0)
    gradcam = GradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=False,
        reshape_transform=reshape_transform,
    )

    # # If targets is None, the highest scoring category will be used for every image in the batch.
    gradcam_output = gradcam(
        input_tensor=image_tensor,
        target_category=None,
        aug_smooth=False,
        eigen_smooth=False,
    )
    original_image = original_image / 255.0

    gradcam_image = show_cam_on_image(
        original_image, gradcam_output[0], use_rgb=False
    )
    return gradcam_image


def streamlit_inference(
    df_test: pd.DataFrame,
    model: Union[models.CustomNeuralNet, Any],
    transform_dict: Dict[str, albumentations.Compose],
    pipeline_config: global_params.PipelineConfig,
    state_dicts: List[OrderedDict],
) -> Dict[str, np.ndarray]:

    """Inference the model and perform TTA, if any.

    Dataset and Dataloader are constructed within this function because of TTA.
    model and transform_dict are passed as arguments to enable inferencing multiple different models.

    Args:
        df_test (pd.DataFrame): The test dataframe.
        model (Union[models.CustomNeuralNet, Any]): The model to be used for inference. Note that pretrained should be set to False.
        transform_dict (Dict[str, albumentations.Compose]): The dictionary of transforms to be used for inference. Should call from get_inference_transforms().
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        state_dicts (List[OrderedDict]): The state dicts of the model.

    Returns:
        all_preds (Dict[str, np.ndarray]): {"normal": normal_preds, "tta": tta_preds}
        mean_tta_preds (np.ndarray): The mean of the tta predictions.
    """

    # a dict to keep track of all predictions [no_tta, tta1, tta2, tta3]
    all_preds = {}
    model = model.to(device)

    # Loop over each TTA transforms, if TTA is none, then loop once over normal inference_augs.
    for aug_name, aug_param in transform_dict.items():
        test_dataset = dataset.CustomDataset(
            df=df_test,
            pipeline_config=pipeline_config,
            transforms=aug_param,
            mode="test",
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **pipeline_config.loader_params.test_loader
        )
        predictions = inference.inference_all_folds(
            model=model,
            state_dicts=state_dicts,
            test_loader=test_loader,
            pipeline_config=pipeline_config,
        )

        all_preds[aug_name] = predictions

    # average over all TTA preds
    mean_tta_score = np.mean(list(all_preds.values()), axis=0)[:, 1]

    return all_preds, mean_tta_score


@st.cache(suppress_st_warning=True)
def load_model(
    model_url: str, model_name: str, save_destination_folder: str
) -> Path:
    """This function loads the model from the model_url and saves it to the save_destination_folder.

    The st.cache decorator is used to cache the model for faster loading.

    Args:
        model_url (str): The url of the model to be loaded.
        model_name (str): The name of the model to be loaded.
        save_destination_folder (str): Path(BASE_DIR, "app/model_weights/tf_efficientnet_b1_ns_5_folds_9qhxwbbq")

    Returns:
        save_destination_folder (Path): Returns the path to the model for use in inference.
    """

    model_filename = model_url.rsplit("/", maxsplit=1)[-1]

    if model_name == "tf_efficientnet_b1_ns":
        # fixed for my experiments
        model_filename_with_uiid = "tf_efficientnet_b1_ns_5_folds_9qhxwbbq"
        model_dir = Path(save_destination_folder / model_filename_with_uiid)
    elif model_name == "resnet50d":
        model_filename_with_uiid = "resnet50d_5_folds_3nvtwvm3"
        model_dir = Path(save_destination_folder / model_filename_with_uiid)

    model_dir.mkdir(exist_ok=True)

    with st.spinner(
        "Downloading model... this may take awhile! \n Don't stop it!"
    ):
        urllib.request.urlretrieve(
            model_url, Path.joinpath(model_dir, model_filename), show_progress
        )

    return model_dir


def get_streamlit_inference_transforms(
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
        )
    }

    return transforms_dict


def main():
    """Main function."""

    # page config
    st.set_page_config(
        page_title="Malignancy Prediction",
        page_icon="💉",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items=None,
    )

    # title
    st.title("Malignancy Scorer and Grad-CAM Interpretation.")

    # image cover
    cover_image = Image.open(
        requests.get(
            "https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/images/cover_page_SIIM-ISIC%20Melanoma%20Classification.png",
            stream=True,
        ).raw
    )
    st.image(cover_image)

    # description
    st.write(
        "This app is designed based off data from SIIM-ISIC Melanoma Classification, 2020. The app will output a malignancy score alongside with its Grad-CAM image for interpretation."
    )

    allowed_extensions = ["jpg", "jpeg", "png"]  # currently allowing jpg only
    ##### PARAMETERS

    # header
    st.header("The Melanoma Scorer")

    # photo upload with argument type for allowed extensions
    melanoma_image = st.file_uploader(
        "1. Upload photo.", type=allowed_extensions
    )
    if melanoma_image is not None:
        image_path = "app/tmp/" + melanoma_image.name
        df_test = pd.DataFrame(data={"image_name": [melanoma_image.name]})
        # no need extension since melanoma_image.name is the name of the file with extension
        df_test["image_path"] = df_test["image_name"].apply(
            lambda x: prepare.return_filepath(
                x,
                folder="app/tmp/",
                extension="",
            )
        )

        # save image to folder
        with open(image_path, "wb") as f:
            f.write(melanoma_image.getbuffer())

        # display pet image
        st.success("Melanoma Image Uploaded Successfully!")

    # model selection
    chosen_model_name = st.selectbox(
        "2. Choose a model.",
        ["EfficientNet B1", "ResNet 50"],
    )
    # @Step 1: Download and load data.
    if chosen_model_name == "EfficientNet B1":
        model_url = "https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/weights/tf_efficientnet_b1_ns_tf_efficientnet_b1_ns_5_folds_9qhxwbbq/tf_efficientnet_b1_ns_best_valid_macro_auroc_fold_1.pt"
        model_name = "tf_efficientnet_b1_ns"
        # override global_params.PipelineConfig so it passes in correctly.
        inference_pipeline_config.model_params.model_name = model_name

    elif chosen_model_name == "ResNet 50":
        # @Step 1: Download and load data.
        model_url = "https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/weights/resnet50d_resnet50d_5_folds_3nvtwvm3/resnet50d_best_valid_macro_auroc_fold_1.pt"
        model_name = "resnet50d"
        inference_pipeline_config.model_params.model_name = model_name

    save_destination_folder = Path(config.BASE_DIR, "app/model_weights")
    model_dir = load_model(model_url, model_name, save_destination_folder)

    # compute malignant score
    if st.button("Compute Malignancy Score"):

        # check if image is uploaded
        if melanoma_image is None:
            st.error("Please upload an image first.")
        else:
            # @Step 2: Inference
            weights = utils.return_list_of_files(
                directory=model_dir, return_string=True, extension=".pt"
            )

            model = models.CustomNeuralNet(
                model_name=model_name,
                out_features=2,
                in_channels=3,
                pretrained=False,
            ).to(device)

            transform_dict = get_streamlit_inference_transforms(
                pipeline_config=inference_pipeline_config,
            )
            # Take note I always save my torch models as .pt files. Note we must return paths as str as torch.load does not support pathlib.
            weights = utils.return_list_of_files(
                directory=model_dir, return_string=True, extension=".pt"
            )

            state_dicts = [
                torch.load(path, map_location=torch.device("cpu"))[
                    "model_state_dict"
                ]
                for path in weights
            ]

            # compute predictions
            with st.spinner("Computing prediction..."):

                # clear memory
                gc.collect()

                predictions, _mean_tta_score = streamlit_inference(
                    df_test=df_test,
                    model=model,
                    transform_dict=transform_dict,
                    pipeline_config=inference_pipeline_config,
                    state_dicts=state_dicts,
                )
                test_predictions = predictions["transforms_test"][0][1]

                # display results
                col1, col2, col3 = st.columns(3)
                melanoma_image = cv2.imread(image_path)
                melanoma_image = cv2.cvtColor(melanoma_image, cv2.COLOR_BGR2RGB)
                reshaped_melanoma_image = cv2.resize(melanoma_image, (256, 256))

                with col1:
                    st.subheader("Original Image")
                    col1.image(reshaped_melanoma_image)

                with col2:
                    st.subheader("Malignance Score")
                    col2.metric(
                        "Malignance Score", f"{test_predictions * 100:.2f}%"
                    )

                    col2.write(
                        "**Note:** The malignance score can be understood as the probability that skin image is malignant."
                    )

                melanoma_image_tensor = get_streamlit_inference_transforms(
                    pipeline_config=inference_pipeline_config
                )["transforms_test"](image=melanoma_image)["image"]

                gradcam_image = streamlit_show_gradcam(
                    model=model,
                    image_tensor=melanoma_image_tensor,
                    original_image=reshaped_melanoma_image,
                    state_dicts=state_dicts,
                    pipeline_config=inference_pipeline_config,
                )

                with col3:
                    st.subheader("Grad-CAM Image")
                    col3.image(gradcam_image)

                # clear memory
                del model, melanoma_image_tensor, gradcam_image, state_dicts
                gc.collect()


def contact_information():
    """Contact information."""

    # The below contact information is referenced from https://github.com/kozodoi/Pet_Pawpularity/blob/main/web_app.py

    # header
    st.header("Contact")

    # website link
    st.write(
        "Check out [my website](https://kozodoi.me) for ML blog, academic publications, Kaggle solutions and more of my work."
    )

    # profile links
    st.write(
        "[![Linkedin](https://img.shields.io/badge/-LinkedIn-306EA8?style=flat&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/reighnss/)](https://www.linkedin.com/in/reighnss/) [![Kaggle](https://img.shields.io/badge/-Kaggle-5DB0DB?style=flat&logo=Kaggle&logoColor=white&link=https://www.kaggle.com/reighns)](https://www.kaggle.com/reighns) [![GitHub](https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white&link=https://github.com/reigHns92)](https://github.com/reigHns92)"
    )

    # copyright
    st.text("© 2022 Hongnan Gao")


main()
contact_information()
