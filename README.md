
- [Intro](#intro)
- [Prerequisites](#prerequisites)
	- [System requirements](#system-requirements)
		- [Hardware requirements](#hardware-requirements)
		- [OS requirements](#os-requirements)
		- [Software requirements](#software-requirements)
	- [Data](#data)
	- [Basic setup:](#basic-setup)
- [Single models](#single-models)
	- [Hyperparameter optimizing training](#hyperparameter-optimizing-training)
	- [Finetuning](#finetuning)
- [Ensemble models](#ensemble-models)
	- [Required single models](#required-single-models)
	- [Setting the best runs](#setting-the-best-runs)
	- [Pre generating the csv splits](#pre-generating-the-csv-splits)
	- [Hyperparameter optimizing training](#hyperparameter-optimizing-training-1)
	- [Visualization](#visualization)

# Intro

This repository contains the code to train and run bone marrow smears classification models used in  diagnosis of Acute Promyelocytic Leukemia from Bone Marrow Smears 

# Prerequisites

## System requirements

The code has been developed and tested mainly on nodes using the IBM-Power9 CPU architecture (ppc64le) with NVIDIA Tesla V100 GPUs provided by the centre for information services and high performance computing (ZIH) TU Dresden. Using similar hardware is recommended for performance and reproducability.

### Hardware requirements

- 1 GPU with at least 8 GB RAM 
- 200 GB disk space
- 64 GB RAM

Hardware requirements are based on [our dataset](#data) and the complete training pipeline. Actual usage heavily depends on the number, cell count, file size and dimension of bone marrow smear images used in training. 

### OS requirements

- OS: Red Hat Enterprise Linux
- Kernel: Linux 4.14.0-115.19.1.el7a.ppc64le
- Architecture: ppc64-le

### Software requirements
- Conda package manager version 4.7.12
- CUDA 10.2.89
- A server running optuna 2.4.0 and mlflow 1.12.1 (Alternativly install it locally via [optional dependencies](https://github.com/TimSchmittmann/Fast-and-Accurate-Diagnosis-of-APL-from-BMS))
- All python dependencies with versions are listed in `conda.environment.yml.example`

## Data

You can download the dataset [from kaggle](https://www.kaggle.com/dataset/a49eb5eb219384adf92856e43dcfc79b9cf1eaea5ec13bd57ef304d173ebe42c). Then unzip it in the root directory of the repository.

If you want to use your own data, you'll need to match the file and csv structure provided in the kaggle dataset. 

## Basic setup:

Clone the repository together with it's submodules:

```
git clone --recurse-submodules git@github.com:TimSchmittmann/Fast-and-Accurate-Diagnosis-of-APL-from-BMS.git
```

Copy environment file to not put your hardware specific setup into version control:

```
cp conda.environment.yml.example environment.yml
```

(optional) Adjust environment.yml to your specific requirements. Current environment.yml will only work on ppc64le architecture. 

Then create and activate the environment:

```
conda env create
conda activate <the_environment>
```

Setup should take no longer than 60 minutes, otherwise you might need to configure the environment.yml with the correct channels for your system. In case of HTTP 500 error for the IBM repository, simply retry until the request is sucessful. 

# Single models

Our framework can be used to train individual models on many different binary classification tasks like Auer rod classification on cell images and APL-Healthy classification on whole BMS images with fully automated hyperparameter optimization. These individual models can then be combined into ensemble models, to improve results even further. To build a strong baseline, it is advised to get familar with the single models first, before trying to construct an ensemble model.

## Hyperparameter optimizing training
We use [automatically registered configuration plugins](https://github.com/TimSchmittmann/bone_marrow_smear_classification/tree/15a946a69e2ffd842358bc38419364a36eea8c72/bms_classification/config/auto_registered/) to configure the automatic training of DL models for individual tasks. You need to create new configs or adjust the existings ones accordingly depending on the task you want to run. For the demo, copy the common_configs.py.example and configure the TrackingConfig depending on your optuna and mlflow setup:

```
cp bone_marrow_smear_classification/bms_classification/config/auto_registered/common_config.py.example bone_marrow_smear_classification/bms_classification/config/auto_registered/common_config.py 
```

This should be enough to run automatically train the demo model on auer rod cells using the "DemoAuerRodCellClassification" configuration. Additionally the PYTHONHASHSEED should be set to 0, to improve caching:

```
PYTHONHASHSEED=0 DISABLE_GPU=False python bone_marrow_smear_classification/bms_classification/model_training/single_model_classification.py --config=DemoAuerRodCellClassification
```

After initial augmentation and creation of bottleneck features, one run should take no longer than 10 minutes.

## Finetuning

After enough runs, you should compare the individual runs on mlflow. It's not recommended to blindly take the best run, as validation loss may oscillate and we should generally prefer simpler models over more complex configurations. Next set the mlflow RUN_ID in the DemoAuerRodCellClassificationFinetuning configuration found in the auer_rod_cell_classification.py file. You may set `LOG_IMAGES` and `LOG_MODELS` to True for finetuning, if you want to create the metrics plots.

# Ensemble models

## Required single models

Creating an ensemble model, requires information about the best hyperparameter configuration for each individual model. Therefore it is required to run the [Hyperparameter optimizing training](#hyperparameter-optimizing-training) configuration for every single model in the ensemble. 

## Setting the best runs

Next the best mlflow runs need to be selected for each single model and set in the auto-registered configuration for the current ensemble. The demo shows the configuration for the APL-healthy classification ensemble and the APL-AML classification ensemble.

## Pre generating the csv splits

Because we must prevent info leaking from train to validation set, we need to pre-split our data into individual cross-validation sets. For the demo, this can be done automatically using

```
python bone_marrow_smear_classification/bms_classification/preparation/ensemble_cv_splits.py
```

For your own data, you should adjust the settings in this python file accordingly.

## Hyperparameter optimizing training

Finally the hyperparameter-optimizing training can be started by utilizing the auto-registered configurations just like with the single models:

```
PYTHONHASHSEED=0 DISABLE_GPU=False python bone_marrow_smear_classification/bms_classification/model_training/ensemble_model_classification.py --config=DemoAplHealthyEnsembleClassification
```

## Visualization

Contrary to the single models there is no finetuning required for the ensemble models, so visualization can be toggled any time using the `LOG_IMAGES` parameter of the CommonTrackingConfiguration. However, to speed up the training it is recommended to leave it turned off and later use the `BEST_TRIAL` setting, to rerun the best trial after enough successful training runs.
