import numpy as np
from cleanlab.filter import find_label_issues

def cleanlab_scrub_labels(X_selected, y_selected_noisy, model):
    """
    Runs Cleanlab Confident Learning to detect and scrub label issues in the selected batch.
    
    Args:
        X_selected (np.ndarray): The selected features from the pool.
        y_selected_noisy (np.ndarray): The corresponding noisy labels.
        model: The trained classifier used to generate out-of-sample predicted probabilities.
        
    Returns:
        clean_indices (np.ndarray): Indices in the selected batch that are deemed clean.
    """
    # Generate predicted probabilities for the selected subset
    pred_probs = model.predict_proba(X_selected)
    
    # Find label issues using Confident Learning
    label_issues = find_label_issues(
        labels=y_selected_noisy,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    # Retain the samples that Cleanlab did NOT flag as corrupted
    clean_indices = np.setdiff1d(np.arange(len(X_selected)), label_issues)
    
    print(f"  Cleanlab detected {len(label_issues)} corrupted labels. Kept {len(clean_indices)} clean samples.")
    
    return clean_indices
