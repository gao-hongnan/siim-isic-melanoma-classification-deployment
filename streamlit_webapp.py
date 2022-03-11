##### PREPARATIONS

# libraries
import gc
import pickle
import os
import sys
import urllib.request
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import torch
from scipy.stats import percentileofscore


# download with progress bar
mybar = None


def show_progress(block_num, block_size, total_size):
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)


##### CONFIG

# page config
st.set_page_config(
    page_title="Score your pet!",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


##### HEADER

# title
st.title("How cute is your pet?")

# image cover
cover_image = Image.open(
    requests.get(
        "https://storage.googleapis.com/kaggle-competitions/kaggle/25383/logos/header.png?t=2021-08-31-18-49-29",
        stream=True,
    ).raw
)
st.image(cover_image)

# description
st.write(
    "This app uses deep learning to estimate a pawpularity score of custom pet photos. Pawpularity is a metric used by [PetFinder](https://petfinder.my/) to judge the pet's attractiveness, which translates to more clicks for the pet profile."
)


##### PARAMETERS

# header
st.header("Score your own pet")

# photo upload
pet_image = st.file_uploader("1. Upload your pet photo.")
if pet_image is not None:

    # check image format
    image_path = "app/tmp/" + pet_image.name
    if (
        (".jpg" not in image_path)
        and (".JPG" not in image_path)
        and (".jpeg" not in image_path)
        and (".bmp" not in image_path)
    ):
        st.error("Please upload .jpeg, .jpg or .bmp file.")
    else:

        # save image to folder
        with open(image_path, "wb") as f:
            f.write(pet_image.getbuffer())

        # display pet image
        st.success("Pet photo uploaded.")

# privacy toogle
choice = st.radio(
    "2. Make the result public?",
    [
        "Yes. Others may see your pet photo.",
        "No. Scoring will be done privately.",
    ],
)

# model selection
model_name = st.selectbox(
    "3. Choose a model for scoring your pet.",
    ["EfficientNet B3", "Swin Transformer"],
)


##### MODELING

# compute pawpularity
if st.button("Compute pawpularity"):

    # check if image is uploaded
    if pet_image is None:
        st.error("Please upload a pet image first.")

    else:

        # specify paths
        if model_name == "EfficientNet B3":
            weight_path = "https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/enet_b3.pth"
            model_path = "app/models/enet_b3/"
        elif model_name == "EfficientNet B5":
            weight_path = "https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/enet_b5.pth"
            model_path = "app/models/enet_b5/"
        elif model_name == "Swin Transformer":
            weight_path = "https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/swin_base.pth"
            model_path = "app/models/swin_base/"

        # download model weights
        if not os.path.isfile(model_path + "pytorch_model.pth"):
            with st.spinner(
                "Downloading model weights. This is done once and can take a minute..."
            ):
                urllib.request.urlretrieve(
                    weight_path, model_path + "pytorch_model.pth", show_progress
                )

        # compute predictions
        with st.spinner("Computing prediction..."):

            # clear memory
            gc.collect()

            # load config
            config = pickle.load(open(model_path + "configuration.pkl", "rb"))

            # initialize model
            model = get_model(
                config, pretrained=model_path + "pytorch_model.pth"
            )
            model.eval()

            # define augmentations
            augs = get_augs(config)

            # process pet image
            pet_image = cv2.imread(image_path)
            pet_image = cv2.cvtColor(pet_image, cv2.COLOR_BGR2RGB)
            image = augs(image=pet_image)["image"]

            # compute prediction
            pred = model(torch.unsqueeze(image, 0))
            score = np.round(100 * pred.detach().numpy()[0][0], 2)

            # compute percentile
            oof = pd.read_csv(model_path + "oof.csv")
            percent = np.round(
                percentileofscore(oof["pred"].values * 100, score), 2
            )

            # display results
            col1, col2 = st.columns(2)
            col1.image(cv2.resize(pet_image, (256, 256)))
            col2.metric("Pawpularity", score)
            col2.metric("Percentile", str(percent) + "%")
            col2.write(
                "**Note:** pawpularity ranges from 0 to 100. Scroll down to read more about the metric and the implemented models."
            )

            # save results
            if choice == "Yes. Others may see your pet photo.":

                # load results
                results = pd.read_csv("app/results.csv")

                # save resized image
                example_img = cv2.imread(image_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_path = "app/images/example_{}.jpg".format(
                    len(results) + 1
                )
                cv2.imwrite(example_path, img=example_img)

                # write score to results
                row = pd.DataFrame(
                    {
                        "path": [example_path],
                        "score": [score],
                        "model": [model_name],
                    }
                )
                results = pd.concat([results, row], axis=0)
                results.to_csv("app/results.csv", index=False)

                # delete old image
                if os.path.isfile(
                    "app/images/example_{}.jpg".format(len(results) - 3)
                ):
                    os.remove(
                        "app/images/example_{}.jpg".format(len(results) - 3)
                    )

            # clear memory
            del config, model, augs, image
            gc.collect()

            # celebrate
            st.success("Well done! Thanks for scoring your pet :)")


##### RESULTS

# header
st.header("Recent results")
with st.expander("See results for three most recently scored pets"):

    # find most recent files
    results = pd.read_csv("app/results.csv")
    if len(results) > 3:
        results = results.tail(3).reset_index(drop=True)

    # display images in columns
    cols = st.columns(len(results))
    for col_idx, col in enumerate(cols):
        with col:
            st.write("**Pawpularity:** ", results["score"][col_idx])
            example_img = cv2.imread(results["path"][col_idx])
            example_img = cv2.resize(example_img, (256, 256))
            st.image(example_img)
            st.write("**Model:** ", results["model"][col_idx])


##### DOCUMENTATION

# header
st.header("More information")

# models
with st.expander("Learn more about the models"):
    st.write(
        "The app uses one of the two computer vision models to score the pet photo. The models are implemented in PyTorch."
    )
    st.table(
        pd.DataFrame(
            {
                "model": ["Swin Transfomer", "EfficientNet B3"],
                "architecture": [
                    "swin_base_patch4_window7_224",
                    "tf_efficientnet_b3_ns",
                ],
                "image size": ["224 x 224", "300 x 300"],
            }
        )
    )

# metric
with st.expander("Learn more about the metric"):
    st.write(
        "Pawpularity is a metric used by [PetFinder.my](https://petfinder.my/), which is a Malaysia's leading animal welfare platform. Pawpularity serves as a proxy for the photo's attractiveness, which translates to more page views for the pet profile.The pawpularity metric ranges from 0 to 100.  Click [here](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/274106) to read more about the metric."
    )

# models
with st.expander("Learn more about image processing"):
    st.write(
        "Before feeding the image to the model, we apply the following transformations:"
    )
    st.write("1. Resizing the image to square shape.")
    st.write("2. Normalizing RGB pixels to ImageNet values.")
    st.write("3. Converting the image to a PyTorch tensor.")


##### CONTACT

# header
st.header("Contact")

# website link
st.write(
    "Check out [my website](https://kozodoi.me) for ML blog, academic publications, Kaggle solutions and more of my work."
)

# profile links
st.write(
    "[![Linkedin](https://img.shields.io/badge/-LinkedIn-306EA8?style=flat&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/kozodoi/)](https://www.linkedin.com/in/kozodoi/) [![Twitter](https://img.shields.io/badge/-Twitter-4B9AE5?style=flat&logo=Twitter&logoColor=white&link=https://www.twitter.com/n_kozodoi)](https://www.twitter.com/n_kozodoi) [![Kaggle](https://img.shields.io/badge/-Kaggle-5DB0DB?style=flat&logo=Kaggle&logoColor=white&link=https://www.kaggle.com/kozodoi)](https://www.kaggle.com/kozodoi) [![GitHub](https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white&link=https://www.github.com/kozodoi)](https://www.github.com/kozodoi) [![Google Scholar](https://img.shields.io/badge/-Google_Scholar-676767?style=flat&logo=google-scholar&logoColor=white&link=https://scholar.google.com/citations?user=58tMuD0AAAAJ&amp;hl=en)](https://scholar.google.com/citations?user=58tMuD0AAAAJ&amp;hl=en) [![ResearchGate](https://img.shields.io/badge/-ResearchGate-59C3B5?style=flat&logo=researchgate&logoColor=white&link=https://www.researchgate.net/profile/Nikita_Kozodoi)](https://www.researchgate.net/profile/Nikita_Kozodoi) [![Tea](https://img.shields.io/badge/-Buy_me_a_tea-yellow?style=flat&logo=buymeacoffee&logoColor=white&link=https://www.buymeacoffee.com/kozodoi)](https://www.buymeacoffee.com/kozodoi)"
)

# copyright
st.text("© 2022 Nikita Kozodoi")
