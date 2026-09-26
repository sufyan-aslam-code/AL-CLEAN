# AL-CLEAN Implementation Audit Report

This report details the implementation status of the core components for the AL-CLEAN project against the requirements defined in the `AL-Clean_Project_Proposal.pdf`.

## 1. Phase 1 (Baseline)
- **Data Preprocessing Pipelines:** **Implemented.** Robust feature scaling using `StandardScaler` for the 'Amount' and 'Time' columns is present in `backend/src/data_loader.py`.
- **Synthetic Label Noise Injection (5% to 30%):** **Implemented.** `inject_label_noise` in `data_loader.py` and `al_clean_pipeline.py` correctly handles flipping binary labels based on a configurable noise level.
- **Baseline XGBoost Initialization:** **Implemented.** `XGBClassifier` is appropriately initialized and used throughout the pipeline.

## 2. Phase 2 (AL/CL Loops)
- **Cleanlab Integration:** **Implemented.** `cleanlab_scrub_labels` in `backend/src/cleaner.py` appropriately utilizes Cleanlab's `find_label_issues` function for confident learning.
- **Active Learning Sampling Loop:** **Implemented.**
  - *Entropy-based metric:* Implemented using Shannon Entropy.
  - *Margin-based uncertainty metric:* Implemented and correctly integrated into the selection pipeline.

## 3. Phase 3 (MLOps)
- **Optuna Hyperparameter Tuning:** **Implemented.** `optimize_xgboost_params` in `backend/src/active_learner.py` properly automates trial optimization.
- **MLflow Tracking:** **Implemented.**
  - `PR-AUC`: Tracked successfully in `backend/src/evaluator.py`.
  - `MCC`: Tracked successfully in `backend/src/evaluator.py`.
  - `F1 Score`: Tracked successfully in `backend/src/evaluator.py`.

## 4. Phase 4 (Validation & UI)
- **Statistical Validation Tests:** **Implemented.** A Wilcoxon signed-rank test is correctly implemented in `backend/src/evaluator.py` (`compute_statistical_significance`).
- **Streamlit Dashboard Structure:** **Implemented.** A Streamlit interactive assessment dashboard is present in `frontend/app.py` to visualize model accuracy degradation versus data-cleaning cycles.

## Verdict
**Ready for Testing**
