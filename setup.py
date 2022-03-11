# Setup installation for the application

from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = [
    "great-expectations==0.13.26",
    "pytest==6.0.2",
    "pytest-cov==2.10.1",
]

dev_packages = [
    "black==20.8b1",
    "pylint==2.11.1",
    "flake8==3.8.3",
    "isort==5.5.3",
    "jupyterlab==2.2.8",
    "pre-commit==2.11.1",
]

docs_packages = [
    "mkdocs==1.1.2",
    "mkdocs-macros-plugin==0.5.0",
    "mkdocs-material==6.2.4",
    "mkdocstrings==0.15.2",
]

setup(
    name="reigHns-cassava",
    version="0.1",
    license="MIT",
    description="Hongnan G's SIIM-ISIC Melanoma Classification Kaggle Competition.",
    author="Hongnan G",
    author_email="reighns.sjr.sjh.gr.twt.hrj@gmail.com",
    url="",
    keywords=["machine-learning", "deep-learning", "scikit-learn"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages + docs_packages,
        "docs": docs_packages,
    },
    dependency_links=[],
    entry_points={
        "console_scripts": [
            "reighns_siim= src.main:app",
        ],
    },
)
