<div align="center">
<h1>SIIM-ISIC Melanoma Classification, 2020</a></h1>
by Hongnan Gao
<br>
</div>

- [Introduction and General Workflow](#introduction-and-general-workflow)
  - [Directory Structure](#directory-structure)
  - [Workflows](#workflows)
  - [Config File](#config-file)
  - [Typical ML-Workflow](#typical-ml-workflow)
- [Clarify the Problem and Constraints](#clarify-the-problem-and-constraints)
- [Understand Your Data Sources](#understand-your-data-sources)
- [Establish Metrics](#establish-metrics)
  - [Benefit Structure](#benefit-structure)
  - [ROC](#roc)
  - [Brier Score Loss](#brier-score-loss)
  - [What could I have done better?](#what-could-i-have-done-better)
- [Explore Data (EDA and Data Inspection/Cleaning)](#explore-data-eda-and-data-inspectioncleaning)
  - [What did EDA and Data Inspection tell us?](#what-did-eda-and-data-inspection-tell-us)
- [Feature Engineering and Preprocessing Steps](#feature-engineering-and-preprocessing-steps)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [What could I have done better?](#what-could-i-have-done-better-1)
- [Cross-Validation Strategy](#cross-validation-strategy)
  - [Step 1: Train-Test-Split](#step-1-train-test-split)
  - [Step 2: Resampling Strategy](#step-2-resampling-strategy)
  - [Cross-Validation Workflow](#cross-validation-workflow)
- [Preliminary Model Selection and Algorithm Spot-Checking](#preliminary-model-selection-and-algorithm-spot-checking)
  - [Preliminary Model Selection (Model Design and Choice) (Importance of EDA)](#preliminary-model-selection-model-design-and-choice-importance-of-eda)
  - [Algorithm Spot-Checking](#algorithm-spot-checking)
    - [Evaluation Results across Folds](#evaluation-results-across-folds)
    - [Naive Error Analysis](#naive-error-analysis)
- [Next Steps](#next-steps)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Retrain on the whole training set](#retrain-on-the-whole-training-set)
  - [Interpretation of Results](#interpretation-of-results)
  - [Benefit Structure and Cost Benefit Analysis](#benefit-structure-and-cost-benefit-analysis)
  - [Model Persistence and Reproducibility](#model-persistence-and-reproducibility)
  - [MLOps and CI/CD](#mlops-and-cicd)


---


## Introduction and General Workflow

This project is about predicting the survivability of coronary artery disease patients.

### Directory Structure

The brief overview of the folder structure is detailed below. Note that `main.py` is the script that `run.sh` will execute. 

```bash
├──data
|    ├──              - raw file
├──config
|    ├── config.py              - configuration file
├──notebooks
|    ├── ...
|    ├── ...
|    ├── ...
|    ├── ...
├──src
|    ├── __init__.py            - make this directory as a Python package
|    ├── metrics_results.py     - functions and classes to store metrics and model results
|    ├── make_folds.py          - make cross-validation folds
|    ├── models.py              - model
|    ├── train.py               - training/optimization pipelines
|    ├── preprocess.py          - preprocessing functions
|    └── utils.py               - utility functions
├── main.py                - main script to call files from src
├── README.md
├── requirements.txt
├── run.sh
```

---

### Workflows


---

### Config File

For small projects and quick prototyping I like to use `dataclass` as my config. For the purpose of the examiners, one can add in scikit-learn's model alongside its hyperparameters in the `classifiers` list below.

The good thing about `dataclass` is I can define methods like `to_yaml` and dump the whole configuration into a `yaml` file should I need to. Furthermore, the `class` allows me to define custom functionalities easily.


```python
@dataclass
class config:
    raw_data: str = "data/survive.db"
    processed_data: str = ""

    seed: int = 1992
    classification_type: str = "binary"
    classifiers = [
        # baseline model
        dummy.DummyClassifier(random_state=1992, strategy="stratified"),
        # linear model
        linear_model.LogisticRegression(random_state=1992, solver="liblinear"),
        # # tree
        tree.DecisionTreeClassifier(random_state=1992),
        # ensemble
        ensemble.RandomForestClassifier(n_estimators=10, random_state=1992),
    ]
    cv_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "cv_schema": "StratifiedGroupKFold",
            "num_folds": 6,
            "train_size": 0.9,
            "shuffle": True,
            "group_kfold_split": "ID",
            "seed": 1992,
        }
    )


    def to_dict(self) -> Dict[str, Any]:
        """Convert the config object to a dictionary.

        Returns:
            Dict: The config object as a dictionary.
        """

        return asdict(self)

    def to_yaml(self, filepath: Union[str, Path]) -> str:
        """Convert the config object to a YAML string and write to file.

        Args:
            filepath (Union[str, Path]): The filepath to write the YAML string to.

        Returns:
            str: The config object as YAML string.
        """
        return yaml.dump(self.to_dict(), filepath)
```

---

### Typical ML-Workflow

The below is a typical ML workflow. We will however follow Nick Singh and Kevin Huo's Ace the Data Science Interview's workflow.

- Data Collection and Ingestion
- Data Extraction (can involve Feature Engineering)
- Data Visualization (understanding data)
- Data Pre-processing
- Model Choice/Design Selection
- Model Training
- Model Evaluation Validation
- Model Interpretability
- Model Deployment and CI/CD

---

## Clarify the Problem and Constraints


## Understand Your Data Sources



## Establish Metrics


### Benefit Structure


### ROC


### Brier Score Loss



### What could I have done better?


## Explore Data (EDA and Data Inspection/Cleaning)



### What did EDA and Data Inspection tell us?



## Feature Engineering and Preprocessing Steps



### Preprocessing Pipeline



### What could I have done better?


## Cross-Validation Strategy


### Step 1: Train-Test-Split


### Step 2: Resampling Strategy



### Cross-Validation Workflow




## Preliminary Model Selection and Algorithm Spot-Checking

### Preliminary Model Selection (Model Design and Choice) (Importance of EDA)


### Algorithm Spot-Checking

#### Evaluation Results across Folds


#### Naive Error Analysis



---

## Next Steps

### Hyperparameter Tuning



### Retrain on the whole training set



### Interpretation of Results



### Benefit Structure and Cost Benefit Analysis



### Model Persistence and Reproducibility



### MLOps and CI/CD



