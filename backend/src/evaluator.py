import numpy as np
import mlflow
from scipy.stats import wilcoxon
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from src.active_learner import active_learning_selection, optimize_xgboost_params
from src.cleaner import cleanlab_scrub_labels

def compute_statistical_significance(standard_scores, al_clean_scores):
    """
    Executes a Wilcoxon signed-rank test comparing the performance vectors across iterations.
    Extracts and returns the p-value.
    """
    try:
        # Wilcoxon test on the paired samples (standard vs AL-clean)
        stat, p_value = wilcoxon(standard_scores, al_clean_scores)
    except ValueError:
        # Fallback if differences are zero across all iterations
        p_value = 1.0
    return p_value

def run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=True, al_step_size=500, iterations=5, metric='entropy'):
    """
    Evaluation loop across active learning iterations.
    Tracks PR-AUC and MCC metrics and logs them to MLflow.
    """
    method_name = "AL-Clean" if use_cleanlab else "Standard_AL"
    print(f"\n--- Starting {method_name.replace('_', ' ')} Evaluation Loop ---")
    
    # Split: ~10% initial train, ~70% pool, ~20% test
    X_train, X_temp, y_train_noisy, y_temp_noisy, y_train_true, y_temp_true = train_test_split(
        X, y_noisy, y_true, test_size=0.9, random_state=42, stratify=y_true
    )
    print(f"[DIAGNOSTIC] Minority class after train_test_split (X_train): {np.sum(y_train_noisy == 1)} noisy labels, {np.sum(y_train_true == 1)} true labels")
    
    X_pool, X_test, y_pool_noisy, y_test_noisy, y_pool_true, y_test_clean = train_test_split(
        X_temp, y_temp_noisy, y_temp_true, test_size=0.22, random_state=42, stratify=y_temp_true
    )
    print(f"[DIAGNOSTIC] Minority class in test set (y_test_clean): {np.sum(y_test_clean == 1)}")
    
    # 0. Optuna Hyperparameter Optimization on Seed Data
    best_params = optimize_xgboost_params(X_train, y_train_noisy, n_trials=5)
    model = XGBClassifier(**best_params)
    
    pr_auc_scores = []
    mcc_scores = []
    f1_scores = []
    total_pruned = 0
    
    for iteration in range(1, iterations + 1):
        # 1. Train the optimized baseline classifier
        model.fit(X_train, y_train_noisy)
        
        # 2. Evaluate generalization on clean test labels
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        mcc = matthews_corrcoef(y_test_clean, preds)
        pr_auc = average_precision_score(y_test_clean, probs)
        f1 = f1_score(y_test_clean, preds)
        
        pr_auc_scores.append(pr_auc)
        mcc_scores.append(mcc)
        f1_scores.append(f1)
        
        print(f"  Iteration {iteration} | MCC: {mcc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Train Size: {len(X_train)}")
        
        # MLflow Metric Tracking
        mlflow.log_metric(f"{method_name}_MCC", mcc, step=iteration)
        mlflow.log_metric(f"{method_name}_PR_AUC", pr_auc, step=iteration)
        mlflow.log_metric(f"{method_name}_F1", f1, step=iteration)
        mlflow.log_metric(f"{method_name}_Train_Size", len(X_train), step=iteration)
        
        # 3. Active Learning Query Selection
        if len(X_pool) < al_step_size:
            print("  Not enough samples left in the pool.")
            break
            
        uncertain_indices = active_learning_selection(model, X_pool, n_samples=al_step_size, metric=metric)
        X_selected = X_pool[uncertain_indices]
        y_selected_noisy = y_pool_noisy[uncertain_indices]
        
        # 4. Label Cleaning via Cleanlab
        if use_cleanlab:
            clean_indices = cleanlab_scrub_labels(X_selected, y_selected_noisy, model)
            total_pruned += (len(X_selected) - len(clean_indices))
        else:
            # Baseline behavior: incorporate all selected noisy data
            clean_indices = np.arange(len(X_selected))
        
        # 5. Integrate approved data into training pool
        X_train = np.vstack([X_train, X_selected[clean_indices]])
        y_train_noisy = np.concatenate([y_train_noisy, y_selected_noisy[clean_indices]])
        
        # 6. Remove acquired indices from unlabeled pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[uncertain_indices] = False
        X_pool = X_pool[mask]
        y_pool_noisy = y_pool_noisy[mask]
        
    return pr_auc_scores, mcc_scores, f1_scores, total_pruned
