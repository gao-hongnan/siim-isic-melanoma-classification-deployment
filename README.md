<div align="center">
<h1>SIIM-ISIC Melanoma Classification, 2020</a></h1>
by Hongnan Gao
<br>
</div>

- [Melanoma Classification with Grad-CAM](#melanoma-classification-with-grad-cam)
- [Competition Description](#competition-description)
- [Directory Structure](#directory-structure)
- [Workflow](#workflow)
- [Grad-CAM](#grad-cam)


---


## Melanoma Classification with Grad-CAM

This repo is an extension of my team's top 4% (ranked 147/3308) solution to the [SIIM-ISIC Melanoma Classification Challenge](https://www.kaggle.com/c/siim-isic-melanoma-classification) held in 2020. 1.5 years have gone by and I decided to deploy the model as a simple webapp with a simple UI consisting of both the predictions and the visualization of the grad-cam.

![cover](https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/images/cover_page_SIIM-ISIC%20Melanoma%20Classification.png)

## Competition Description

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.

Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma. Existing AI approaches have not adequately considered this clinical frame of reference. Dermatologists could enhance their diagnostic accuracy if detection algorithms take into account “contextual” images within the same patient to determine which images represent a melanoma. If successful, classifiers would be more accurate and could better support dermatological clinic work.

As the leading healthcare organization for informatics in medical imaging, the Society for Imaging Informatics in Medicine (SIIM)'s mission is to advance medical imaging informatics through education, research, and innovation in a multi-disciplinary community. SIIM is joined by the International Skin Imaging Collaboration (ISIC), an international effort to improve melanoma diagnosis. The ISIC Archive contains the largest publicly available collection of quality-controlled dermoscopic images of skin lesions.

In this competition, you’ll identify melanoma in images of skin lesions. In particular, you’ll use images within the same patient and determine which are likely to represent a melanoma. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.

Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people. - [SIIM-ISIC Melanoma Classification Challenge](https://www.kaggle.com/c/siim-isic-melanoma-classification)

## Directory Structure

Note that most of the scripts in this repo is used for training, for the purpose of deployment, we just need to concentrade on the below:

```python
streamlit_app.py
app/
```

`streamlit_app.py` is the main entry point of the webapp and is almost self-contained, with some dependencies on my `src/` codes. `app/` is the main directory of the webapp with weights and images stored.

---

## Workflow

`requirements.txt` is the list of dependencies for the webapp, both suitable for training and deployment. 

To setup the environment, we can do the following:

```bash
# Assuming Windows
python -m venv venv_streamlit
venv_streamlit\Scripts\activate
python -m pip install --upgrade pip setuptools wheel # upgrade pip
pip install -e .  # installs required packages only    
```

A word of caution however, is that I removed all `+cu` from the `requirements.txt` file. This is because I have not yet figured out how to install the **CUDA** version of the `pytorch` package in **Streamlit Cloud**. I believe this is a known issue and should be on their future roadmap. Consequently, if one wants to train models using my scripts, it is best to go my [main repo](https://github.com/reigHns92/siim-isic-melanoma-classification) to do so.

---

## Grad-CAM

I have been working on Grad-CAM for a while now, and I have a [tutorial](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/computer_vision/general/neural_network_interpretation/05_gradcam_and_variants/gradcam_explained/) on it. The [paper](https://arxiv.org/abs/1610.02391) is the original paper and the implementation in python is [here](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/computer_vision/general/neural_network_interpretation/05_gradcam_and_variants/gradcam_from_scratch/).