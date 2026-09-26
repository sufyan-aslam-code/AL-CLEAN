from core.pipeline import run_full_pipeline
from core.data_loader import load_kaggle_fraud_data, inject_label_noise
from core.evaluator import run_evaluation_loop

def test_determinism():
    X, y_true = load_kaggle_fraud_data(csv_path="data/creditcard.csv", subsample=1000)
    
    # Run 1
    y_noisy1 = inject_label_noise(y_true, noise_level=0.15)
    pr1, mcc1, _ = run_evaluation_loop(X, y_noisy1, y_true, use_cleanlab=True, al_step_size=50, iterations=2, metric='entropy')
    
    # Run 2
    y_noisy2 = inject_label_noise(y_true, noise_level=0.15)
    pr2, mcc2, _ = run_evaluation_loop(X, y_noisy2, y_true, use_cleanlab=True, al_step_size=50, iterations=2, metric='entropy')
    
    print(f"Run 1 PR-AUC: {pr1}")
    print(f"Run 2 PR-AUC: {pr2}")
    
    if pr1 == pr2:
        print("Determinism Verified! Both runs yielded exactly identical arrays.")
    else:
        print("Determinism Failed. Arrays differ.")

if __name__ == '__main__':
    test_determinism()
