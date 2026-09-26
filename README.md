<div align="center">
  
# 🧼 AL-Clean

**An End-to-End Active Learning Microservice Framework for Heavily Skewed & Noisy Tabular Data**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-orange)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)](https://mlflow.org/)
[![Cleanlab](https://img.shields.io/badge/Cleanlab-Confident_Learning-black)](https://cleanlab.ai/)

---
</div>

## 📖 Overview

**AL-Clean** is a production-grade machine learning framework built to tackle two of the most pernicious problems in real-world data science: **Extreme Class Imbalance** and **Label Noise**. 

Rather than relying on naive synthetic oversampling (like SMOTE) which often exacerbates model degradation by hallucinating synthetic noise, AL-Clean introduces a rigorous **data-purification loop** coupled with dynamic **Shannon Entropy uncertainty sampling**. 

---

## 🚨 The Core Problem

Real-world datasets (such as fraud detection, medical diagnosis, or anomaly detection) are rarely clean or balanced. 

- 📉 **Extreme Imbalance:** The target minority class often constitutes a minuscule fraction of the data (e.g., ~99.8% imbalance).
- 🧬 **Label Noise:** Data annotated by humans or weak heuristics is frequently mislabeled, containing anywhere from 15% to 30% noise.
- ❌ **The SMOTE Trap:** Traditional synthetic oversampling amplifies this noise, confusing decision boundaries.

**The Solution?** Query only the most informative data, and mathematically scrub out the corrupted labels before they ever touch the training pool.

---

## ⚡ Key Technical Innovations

* 🧠 **Dynamic Uncertainty Sampling:** Utilizes **Shannon Entropy** to dynamically query the top-K most uncertain samples per iteration.
* 🧼 **Confident Learning:** Integrates **Cleanlab** into a real-time data-purification loop, identifying and scrubbing noisy labels out of the actively queried batch.
* 🚀 **XGBoost & Optuna:** Leverages **XGBoost** as the primary classification engine to generate reliable posterior probability distributions, with hyperparameter tuning automated via **Optuna**.
* 📊 **Threshold-Independent Evaluation:** Evaluates system robustness strictly using **PR-AUC** (Precision-Recall Area Under Curve) and **MCC** (Matthews Correlation Coefficient) to prevent the masking of minority class performance.
* 📈 **Live Budget Tracking:** Features an interactive **Streamlit dashboard** that dynamically tracks cumulative annotation budget savings.

---

## 🏗️ Decoupled Architecture

The repository is built strictly as a decoupled microservice architecture:

```text
AL-CLEAN/
├── api/                  # 🌐 FastAPI execution bridge and request schemas
├── core/                 # 🧠 Core ML Logic (Pipeline, XGBoost, Entropy Sampler, Cleanlab)
├── dashboard/            # 📊 Streamlit interactive UI (Budget tracking & metrics)
├── tests/                # 🧪 Automated determinism testing suite
├── data/                 # 📂 Local data storage (e.g., creditcard.csv)
├── requirements.txt      # 📦 Dependency management
└── README.md             # 📝 Documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/AL-Clean.git
cd AL-Clean
pip install -r requirements.txt
```

### 2. Launch the Microservices

The framework is divided into three distinct services. We recommend running each in a separate terminal instance.

#### 📈 A. Start MLflow Tracking Server
Tracks all active learning iterations, parameters, and metrics dynamically.
```bash
mlflow ui --port 5000
```
*Access at: `http://localhost:5000`*

#### ⚙️ B. Start the FastAPI Backend
The API bridge that orchestrates the data pipeline and machine learning loops.
```bash
uvicorn api.main:app --reload --port 8000
```
*Access at: `http://localhost:8000/docs` (Swagger UI)*

#### 🎨 C. Start the Streamlit Dashboard
An elegant UI to trigger pipelines, adjust label noise configurations, and visualize PR-AUC, MCC, and Budget Savings.
```bash
streamlit run dashboard/app.py
```
*Access at: `http://localhost:8501`*

---

## 🧪 Automated Testing

Ensure the determinism and integrity of the core pipeline by running the automated test suite from the root directory:

```bash
python -m tests.test_determinism
```

---

<div align="center">
  <i>Built with ❤️ for rigorous Machine Learning Engineering.</i>
</div>
