# AL-Clean: Uncertainty-Aware Active Learning for Cost-Efficient Data Labeling

## Abstract
Active learning (AL) is a powerful paradigm for reducing data annotation costs by strategically selecting the most informative samples for a model to learn from. However, standard active learning strategies implicitly assume the presence of a flawless oracle. In real-world scenarios where human annotators are fallible, labeling noise is inevitable. This label noise disproportionately degrades active learning because uncertainty sampling algorithms inherently prioritize ambiguous, hard-to-classify examples—which are precisely the samples most likely to be mislabeled by annotators. As a result, the active learning loop actively poisons its own training set. **AL-Clean** introduces a robust, cost-efficient framework that integrates Confident Learning (via the Cleanlab library) directly into the active learning acquisition loop. By automatically auditing the chosen high-uncertainty samples and systematically filtering out label errors before they enter the training pool, AL-Clean ensures robust model convergence and maintains data-efficiency even in highly corrupted environments.

## Mathematical Formulation
In our pipeline, the informativeness of an unlabeled instance $x$ is quantified using **Shannon Entropy**. For a binary classification task, the model outputs predicted probabilities $P(y=1|x)$ and $P(y=0|x)$. The entropy $H(x)$ is calculated as:

$$
H(x) = - \sum_{c \in \{0, 1\}} P(y=c|x) \log P(y=c|x)
$$

The active learning acquisition function selects a batch of $N$ samples that maximize this uncertainty metric:

$$
X_{selected} = \underset{X_{batch} \subset X_{pool}, |X_{batch}|=N}{\arg\max} \sum_{x \in X_{batch}} H(x)
$$

These selected samples are subsequently evaluated by Cleanlab's Confident Learning filter, which computes the joint distribution of noisy labels and true labels to identify and discard instances with critical label issues.

## Experimental Design & MLOps Architecture
To empirically validate our approach, we established a modular, production-grade protocol:
* **Dataset**: We leverage the real-world **Kaggle Credit Card Fraud Detection Dataset**, preserving its extreme class imbalance (99.8% Normal, 0.2% Fraud).
* **Label Corruption**: We artificially inject a **15% label noise** into the target labels to simulate highly unreliable human annotation.
* **Hyperparameter Optimization (Optuna)**: Before initiating active learning, the initial seed dataset undergoes a 5-trial Optuna optimization to tune XGBoost parameters (`max_depth`, `learning_rate`, `n_estimators`, `scale_pos_weight`).
* **Active Learning Loop**: The pipeline executes over 5 iterations. In each round, the XGBoost model acquires $N=500$ new samples using Shannon Entropy.
* **Intervention**: 
  * *Standard AL*: Blindly accepts all 500 queried samples into the training set.
  * *AL-Clean*: Passes the 500 samples through Cleanlab to scrub mislabeled instances prior to integration.
* **Experiment Tracking (MLflow)**: Execution parameters, PR-AUC, Matthews Correlation Coefficient (MCC) metrics, and benchmark artifacts are automatically captured and logged to a local MLflow registry.
* **Interactive Frontend (Streamlit)**: A web dashboard to dynamically visualize benchmark metrics, test statistical significance (Wilcoxon Signed-Rank Test), and observe AL-Clean's impact in real-time.

## Project Structure
```text
al-clean-project/
├── src/
│   ├── data_loader.py    # Loads Kaggle dataset & injects corruption
│   ├── active_learner.py # Optuna tuning & Shannon Entropy acquisition
│   ├── cleaner.py        # Cleanlab Confident Learning wrapper
│   └── evaluator.py      # Core evaluation loop & metric tracking
├── main.py               # MLOps benchmark execution (MLflow entry point)
├── app.py                # Streamlit dashboard interface
├── requirements.txt      # Pinned Python dependencies
└── README.md
```

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Dataset:**
   Download the `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root of this project repository.

3. **Execute the Core Benchmark Pipeline (with MLflow):**
   ```bash
   python main.py
   ```
   To view the experiment logs, spin up the MLflow UI:
   ```bash
   mlflow ui
   ```

4. **Launch the Interactive Dashboard:**
   To explore the results visually and adjust noise configurations dynamically, run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
