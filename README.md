# AL-Clean

AL-Clean is an end-to-end active learning microservice framework designed to handle heavily skewed tabular data with injected label noise, strictly avoiding synthetic oversampling methods like SMOTE.

## The Core Problem

In many real-world machine learning scenarios, such as fraud detection, datasets suffer from:
1. **Extreme Class Imbalance:** The minority class (e.g., fraud) makes up a minuscule fraction of the data (~99.8% class imbalance).
2. **Label Noise:** Data is often mislabeled by human annotators or weak heuristics (15%-30% noise).

Traditional approaches rely on synthetic oversampling (SMOTE), which often exacerbates the issue when trained on noisy labels by creating synthetic noise. AL-Clean solves this through a robust data-purification loop combined with uncertainty sampling.

## Architecture

The codebase strictly adheres to a decoupled microservice layout:

- `api/`: FastAPI application and request/response schemas serving as the execution bridge.
- `core/`: Core machine learning logic, containing:
  - XGBoost trainer
  - Shannon Entropy uncertainty sampler
  - Cleanlab noise cleaner
  - PR-AUC and MCC evaluator
- `dashboard/`: Interactive Streamlit UI for visual assessment and budget savings tracking.
- `tests/`: Automated testing scripts to ensure pipeline determinism.

## Key Technical Components

- **Shannon Entropy Uncertainty Sampling**: Dynamically queries the top-K most uncertain samples per iteration.
- **Cleanlab Confident Learning**: Integrates a real-time data-purification loop to identify and scrub noisy labels out of the queried batch.
- **XGBoost & Optuna**: Leverages XGBoost as the primary classification engine to generate reliable posterior probability distributions, optimized dynamically using Optuna.
- **MLflow Tracking**: Production-grade execution bridge for tracking experiments, metrics, and parameters.
- **Strict Evaluation**: System robustness is evaluated exclusively using threshold-independent metrics (PR-AUC and MCC) due to the severe class imbalance.

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Services

1. **Start MLflow Tracking Server**:
   ```bash
   mlflow ui --port 5000
   ```
   The tracking server will be available at `http://localhost:5000`.

2. **Start the FastAPI Backend**:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   The API will be available at `http://localhost:8000`.

3. **Start the Streamlit Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   The interactive UI will be available at `http://localhost:8501`.

From the Streamlit UI, you can easily configure the noise level, active learning iterations, and sampling metrics, and observe the cumulative annotation budget savings!
