import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

def optimize_xgboost_params(X_train, y_train, n_trials=5):
    """
    Optimizes XGBoost hyperparameters using Optuna on the initial seed dataset.
    """
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # Quick cross-validation to assess trial parameters
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = XGBClassifier(**param)
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict_proba(X_train[val_idx])[:, 1]
            scores.append(average_precision_score(y_train[val_idx], preds))
            
        return np.mean(scores)
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    print(f"  Starting Optuna optimization for {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    print(f"  Optuna optimization finished. Best params: {best_params}")
    
    return best_params

def active_learning_selection(model, X_pool, n_samples=500, metric='entropy'):
    """
    Selects the most uncertain samples using Shannon Entropy or Margin.
    
    Args:
        model: The trained classifier capable of predict_proba.
        X_pool (np.ndarray): The unlabeled pool features.
        n_samples (int): Number of samples to select.
        metric (str): 'entropy' or 'margin'.
        
    Returns:
        uncertain_indices (np.ndarray): Indices of the selected top N most uncertain samples.
    """
    probs = model.predict_proba(X_pool)
    
    # Small epsilon to prevent log(0) mathematically
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)
    
    if metric == 'entropy':
        # Compute Shannon Entropy: -\sum p * log(p)
        uncertainty = -np.sum(probs * np.log(probs), axis=1)
        uncertain_indices = np.argsort(uncertainty)[::-1][:n_samples]
    elif metric == 'margin':
        # Margin: |P(class 0) - P(class 1)|
        margin = np.abs(probs[:, 0] - probs[:, 1])
        # Smallest margin = most uncertain
        uncertain_indices = np.argsort(margin)[:n_samples]
    else:
        raise ValueError("metric must be 'entropy' or 'margin'")
    
    return uncertain_indices
