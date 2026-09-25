import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from cleanlab.filter import find_label_issues
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split

def generate_simulated_fraud_data(n_samples=50000, weights=(0.98, 0.02)):
    """
    Generates an imbalanced dataset simulating fraud detection.
    """
    print(f"Generating simulated fraud data with {n_samples} samples and {weights} class distribution...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=list(weights),
        flip_y=0, # we inject noise manually
        random_state=42
    )
    return X, y

def inject_label_noise(y_train, noise_level=0.15):
    """
    Intentionally flips a given percentage of binary labels to simulate data corruption.
    """
    print(f"Injecting {noise_level*100}% label noise...")
    y_train_noisy = y_train.copy()
    n_samples = len(y_train)
    n_noise = int(noise_level * n_samples)
    
    # Select random indices to flip
    noise_indices = np.random.choice(n_samples, size=n_noise, replace=False)
    
    # Flip the binary labels (0 becomes 1, 1 becomes 0)
    y_train_noisy[noise_indices] = 1 - y_train_noisy[noise_indices]
    
    return y_train_noisy

def active_learning_selection(model, X_pool, n_samples=500):
    """
    Phase 2: Selects the most uncertain samples using Shannon Entropy.
    """
    probs = model.predict_proba(X_pool)
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)
    
    # Compute Shannon Entropy: -\sum p * log(p)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    
    # Top N indices with highest entropy (most uncertain)
    uncertain_indices = np.argsort(entropy)[::-1][:n_samples]
    return uncertain_indices

def cleanlab_scrub_labels(X_selected, y_selected_noisy, model):
    """
    Phase 3: Runs Cleanlab Confident Learning to detect and scrub label issues.
    """
    # For robust confident learning we need out-of-sample probabilities, 
    # but here we use the model's current predictions as a proxy.
    pred_probs = model.predict_proba(X_selected)
    
    label_issues = find_label_issues(
        labels=y_selected_noisy,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    # Keep the samples that Cleanlab did NOT flag as label issues
    clean_indices = np.setdiff1d(np.arange(len(X_selected)), label_issues)
    print(f"  Cleanlab detected {len(label_issues)} corrupted labels. Kept {len(clean_indices)} clean samples.")
    return clean_indices

def run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=True):
    """
    Phase 2 & 3 Pipeline: Evaluation loop across 5 AL iterations.
    If use_cleanlab is True, applies Cleanlab filtering. Otherwise, adds all selected samples blindly.
    Returns a list of PR-AUC scores across iterations.
    """
    method_name = "AL-Clean (With Cleanlab)" if use_cleanlab else "Standard AL (No Cleanlab)"
    print(f"\n--- Starting {method_name} Evaluation Loop ---")
    
    # Initial split: ~10% initial train, ~70% pool, ~20% test
    X_train, X_temp, y_train_noisy, y_temp_noisy, y_train_true, y_temp_true = train_test_split(
        X, y_noisy, y_true, test_size=0.9, random_state=42, stratify=y_true
    )
    X_pool, X_test, y_pool_noisy, y_test, _, _ = train_test_split(
        X_temp, y_temp_noisy, y_temp_true, test_size=0.22, random_state=42 
    )
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    pr_auc_scores = []
    
    for iteration in range(1, 6):
        # Train baseline model
        model.fit(X_train, y_train_noisy)
        
        # Evaluate against the clean true labels
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        pr_auc = average_precision_score(y_test, probs)
        
        pr_auc_scores.append(pr_auc)
        print(f"  Iteration {iteration} | Test Accuracy: {acc:.4f} | PR-AUC: {pr_auc:.4f} | Train Size: {len(X_train)}")
        
        # Active Learning Selection
        if len(X_pool) < 500:
            print("  Not enough samples left in the pool.")
            break
            
        uncertain_indices = active_learning_selection(model, X_pool, n_samples=500)
        X_selected = X_pool[uncertain_indices]
        y_selected_noisy = y_pool_noisy[uncertain_indices]
        
        # Cleanlab Integration
        if use_cleanlab:
            clean_indices = cleanlab_scrub_labels(X_selected, y_selected_noisy, model)
        else:
            # Blindly accept all uncertain samples if Cleanlab is skipped
            clean_indices = np.arange(len(X_selected))
        
        # Add the data back to the training set
        X_train = np.vstack([X_train, X_selected[clean_indices]])
        y_train_noisy = np.concatenate([y_train_noisy, y_selected_noisy[clean_indices]])
        
        # Remove selected samples from the unlabeled pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[uncertain_indices] = False
        X_pool = X_pool[mask]
        y_pool_noisy = y_pool_noisy[mask]
        
    return pr_auc_scores

if __name__ == "__main__":
    # Phase 1: Data Generation and Noise Injection
    X, y_true = generate_simulated_fraud_data()
    y_noisy = inject_label_noise(y_true, noise_level=0.15)
    
    # Phase 2 & 3: Run the AL + Cleanlab Pipeline
    clean_scores = run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=True)
    
    # Phase 4: Run Standard Active Learning Pipeline for Benchmarking
    standard_scores = run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=False)
    
    # Output Benchmarking Table
    print("\n" + "="*55)
    print("PHASE 4: BENCHMARK RESULTS (PR-AUC Scores)")
    print("="*55)
    print("| Iteration | Standard AL (No Cleanlab) | AL-Clean (With Cleanlab) |")
    print("|-----------|---------------------------|--------------------------|")
    for i, (std, cln) in enumerate(zip(standard_scores, clean_scores), 1):
        print(f"| {i:<9} | {std:<25.4f} | {cln:<24.4f} |")
    print("="*55)
    
    print("\nPipeline completed successfully.")
