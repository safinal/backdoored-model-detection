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
Our approach leverages **activation-based optimization** and **statistical anomaly detection** to identify whether a given image classification model contains a backdoor. Here’s an overview of the method:

### 1. **Activation Inversion Optimization**

For each class in the dataset:

- We select a random batch of clean images and pass them through the first convolutional and residual layers of the model to obtain activations.
- Then, treating these activations as trainable variables, we **optimize them** (using the model’s latter layers held fixed) so that the final output is maximally confident towards a *chosen target class*.
- The process simulates searching for possible hidden triggers in the model’s representation that could elicit a strong, suspiciously focused output for any class.

### 2. **Maximal Confidence Statistic**

- For each class, we compute a **maximal confidence score**:  
  The model's maximum output (for the target class) after the activation inversion process, penalized for high outputs in other classes.
- This yields a **vector of scores**, one per class.

### 3. **Statistical Outlier Detection**

- If the model is clean, these maximal confidence values are usually balanced across all classes.
- A backdoored model, however, often contains one class whose maximal confidence is an **outlier** (because the backdoor trigger creates unusually high confidence for the target class).
- We fit an **exponential distribution** to the scores *excluding* the maximum. We then calculate the **p-value** of the actual maximum under this distribution to measure how extreme it is.
- If this p-value is below a set threshold (e.g., 0.08), the model is flagged as containing a backdoor.

### 4. **Key Features**

- **No dependence on knowledge of the specific trigger or attack type.**
- **Relies only on a small set of clean images and the model**—aligned with problem constraints.
- **Principled statistical decision**: robust against noise and overfitting.

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
