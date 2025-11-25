# ⭐️ Backdoored Model Detection

[![pytorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔍 Overview
This repository contains our solution for the Backdoored Model Detection Challenge, part of the [**Rayan International AI Contest**](https://ai.rayan.global). The challenge aims to develop a system that classifies models as either backdoored or clean.

## 🎯 Challenge Objective
The goal is to build a robust system that can:

- **Process input consisting of:**
  - A machine learning **model** (specifically, a PreActResNet18 model trained on an image classification task)
  - The **number of classes** in the dataset
  - A **folder path to clean test images** (representing 1% of the test dataset)
  - The **image transformation** function used in model preprocessing

- **Identify and return** whether the provided model contains a backdoor, with output:
  - `0` if the model is **backdoored**
  - `1` if the model is **clean**

For more details, check the `problem_description.pdf` file.

## ⚙️ Constraints

- **🗂️ Maximum Upload File Size:** 1MB  

- **🔢 Constant Parameters Limitation:**  
  - ❌ No more than one constant parameter (e.g., a threshold to discriminate models trained on the evaluation set)  

- **🌐 Internet Access:**  
  - ❌ No internet access will be granted during the test process  

- **⏱️ Computation Time Per Sample:**  
  - ⏳ 60-second limit per data sample for answer computation  

- **📁 File Permissions:**  
  - ❌ Strictly prohibited from reading or writing any files in your solution


## 🧠 Our Approach
Our solution extends the **Mm-Bd** method by introducing feature-space optimization, improved initialization, image resizing, and a refined statistical detection rule. We address the challenge of detecting backdoors without prior knowledge of the attack type or trigger structure by reverse-engineering the model's sensitivity in the feature space.

### 1. Activation Perturbation Optimization
We attempt to expose latent triggers by optimizing activation patterns to maximize class confidence.
* **Initialization:** We sample clean images from the verification set and resize them to $128\times128$, as this resolution consistently improves accuracy.
* **Feature Extraction:** Inputs are passed through the model up to `model.layer1[0]` to produce an initial feature tensor. We found that initializing from real features yields better results than random initialization.
* **Optimization:** We freeze deeper layers and treat the extracted activations as trainable variables. We optimize these activations to maximize the model's confidence for a target class.
* **Scoring:** We compute a maximal confidence score for each class, penalized by the activation magnitudes of other classes to ensure specificity.

### 2. Statistical Outlier Detection
We employ a statistical anomaly test to distinguish between clean and poisoned models.
* **Hypothesis:** Clean models exhibit balanced confidence across classes, whereas backdoored models display unusually high maximal confidence for a specific target class.
* **Distribution Fitting:** We fit the maximal confidence scores (excluding the highest one) to an **exponential distribution**, which provided the best separation in our experiments.
* **Decision Rule:** We treat the maximum score as a candidate outlier and calculate its p-value. If the **p-value is below 0.08**, the model is classified as **backdoored (`0`)**; otherwise, it is classified as **clean (`1`)**.

## 🏆 Results

Our solution for this Challenge achieved outstanding results. The evaluation metric for this challenge is **Accuracy**, with submissions tested on a private test dataset. Our method achieved the **second highest score**.

The table below presents a summary of the Top 🔟 teams and their respective accuracy scores:

| **Rank** | **Team**                             | **Accuracy (%)** |
|----------|--------------------------------------|------------------|
|🥇        | AUTs                                 | 79            |
|🥈        | **No Trust Issues Here (Our Team)**  | **78**        |
|🥉        | Persistence                          | 74            |
| 4        | AI Guardians of Trust                | 72            |
| 5        | AIUoK                                | 70            |
| 6        | Pileh                                | 67            |
| 7        | My Team                              | 66            |
| 8        | Unknown                              | 66            |
| 9        | red_serotonin                        | 65            |
| 10       | DevNull                              | 65            |


## 🏃🏻‍♂️‍➡️ Steps to Set Up and Run

Follow these instructions to set up your environment and execute the pipeline.

### 1. Clone the Repository
```bash
git clone git@github.com:safinal/backdoored-model-detection.git
cd backdoored-model-detection
```
### 2. Set Up the Environment
We recommend using a virtual environment to manage dependencies.

Using ```venv```:
```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```
Using ```conda```:
```bash
conda create --name backdoored-model-detection python=3.8 -y
conda activate backdoored-model-detection
```
### 3. Install Dependencies
Install all required libraries from the ```requirements.txt``` file:
```bash
pip install -r requirements.txt
```
### 4. Run
```bash
python run.py --config ./config/config.yaml
```

## 🤝🏼 Contributions
We welcome contributions from the community to make this repository better!
