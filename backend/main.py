import mlflow
from src.data_loader import load_kaggle_fraud_data, inject_label_noise
from src.evaluator import run_evaluation_loop

def main():
    print("=======================================================")
    print("AL-Clean: Modular Pipeline Execution (Kaggle Dataset)")
    print("=======================================================\n")
    
    noise_level = 0.15
    al_step_size = 500
    
    with mlflow.start_run(run_name="AL-Clean Phase 3 Execution"):
        # Log Pipeline Parameters
        mlflow.log_param("noise_level", noise_level)
        mlflow.log_param("al_step_size", al_step_size)
        mlflow.log_param("dataset", "creditcard.csv")
        
        # 1. Dataset Loading & Corruption Injection
        X, y_true = load_kaggle_fraud_data(csv_path="../data/creditcard.csv")
        y_noisy = inject_label_noise(y_true, noise_level=noise_level)
        
        # 2. Execute AL-Clean Pipeline (With Scrubbing)
        clean_pr, clean_mcc, clean_f1, clean_pruned = run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=True, al_step_size=al_step_size)
        
        # 3. Execute Baseline Pipeline (Standard Uncertainty Sampling)
        standard_pr, standard_mcc, standard_f1, _ = run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=False, al_step_size=al_step_size)
        
        # 4. Compile Markdown Table
        markdown_table = (
            "| Iteration | Standard AL (PR-AUC | MCC | F1) | AL-Clean (PR-AUC | MCC | F1) |\n"
            "|-----------|---------------------------------|---------------------------------|\n"
        )
        for i in range(len(standard_pr)):
            markdown_table += f"| {i+1:<9} | {standard_pr[i]:.4f} | {standard_mcc[i]:.4f} | {standard_f1[i]:.4f} | {clean_pr[i]:.4f} | {clean_mcc[i]:.4f} | {clean_f1[i]:.4f} |\n"
            
        print("\n" + "="*77)
        print("PHASE 4: BENCHMARK RESULTS (PR-AUC & MCC Scores)")
        print("="*77)
        print(markdown_table, end="")
        print("="*77)
        
        # Log the markdown table as an MLflow artifact
        mlflow.log_text(markdown_table, "benchmark_results.md")
        
    print("\nModular Pipeline completed successfully.")

if __name__ == "__main__":
    main()
