import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_kaggle_fraud_data(csv_path="creditcard.csv", subsample=None):
    """
    Loads and preprocesses the real-world Kaggle Credit Card Fraud Detection dataset.
    Provides download instructions if the dataset is not found.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Dataset '{csv_path}' not found in the root directory.")
        print("Please download the 'Credit Card Fraud Detection' dataset from Kaggle:")
        print("URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"Extract the archive and ensure '{csv_path}' is placed here.")
        sys.exit(1)
        
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Robust Feature Scaling (Cleanly scaling Time and Amount)
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    # Optional subsampling for faster iterative testing
    if subsample is not None and subsample < len(df):
        print(f"Subsampling dataset to {subsample} rows...")
        np.random.seed(42)
        indices = np.random.choice(len(df), size=subsample, replace=False)
        X = X[indices]
        y = y[indices]
        
    class_1_pct = (np.sum(y == 1) / len(y)) * 100
    print(f"Dataset loaded: {X.shape[0]} samples. Extreme Imbalance Fraud Class Distribution: {class_1_pct:.3f}%")
    
    return X, y

def inject_label_noise(y_train, noise_level=0.15):
    """
    Intentionally flips a given percentage of binary labels to simulate data corruption.
    
    Args:
        y_train (np.ndarray): The original labels.
        noise_level (float): The percentage of labels to flip (0.0 to 1.0).
        
    Returns:
        y_train_noisy (np.ndarray): The labels with injected noise.
    """
    print(f"Injecting {noise_level*100}% label noise...")
    y_train_noisy = y_train.copy()
    n_samples = len(y_train)
    n_noise = int(noise_level * n_samples)
    
    # Select random indices to flip
    np.random.seed(42) # Fixed seed for reproducibility
    noise_indices = np.random.choice(n_samples, size=n_noise, replace=False)
    
    # Flip the binary labels (0 becomes 1, 1 becomes 0)
    y_train_noisy[noise_indices] = 1 - y_train_noisy[noise_indices]
    
    return y_train_noisy
