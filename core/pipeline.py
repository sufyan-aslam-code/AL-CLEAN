import mlflow
from core.data_loader import load_kaggle_fraud_data, inject_label_noise
from core.evaluator import run_evaluation_loop

def run_full_pipeline(noise_level=0.15, iterations=5, metric='entropy', al_step_size=500, run_id=None):
    print("=======================================================")
    print("AL-Clean: Modular Pipeline Execution (Kaggle Dataset)")
    print("=======================================================\n")
    
    mlflow.set_tracking_uri("http://localhost:5000")
    
    if run_id:
        # Resume the run created by the API
        run_context = mlflow.start_run(run_id=run_id)
    else:
        run_context = mlflow.start_run(run_name="AL-Clean Phase 3 Execution")
        
    with run_context as run:
        # Log Pipeline Parameters
        mlflow.log_param("noise_level", noise_level)
        mlflow.log_param("al_step_size", al_step_size)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("sampling_metric", metric)
        mlflow.log_param("dataset", "creditcard.csv")
        
        # 1. Dataset Loading & Corruption Injection
        X, y_true = load_kaggle_fraud_data(csv_path="data/creditcard.csv")
        y_noisy = inject_label_noise(y_true, noise_level=noise_level)
        
        # 2. Execute AL-Clean Pipeline (With Scrubbing)
        clean_pr, clean_mcc, clean_pruned = run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=True, al_step_size=al_step_size, iterations=iterations, metric=metric)
        
        # 3. Execute Baseline Pipeline (Standard Uncertainty Sampling)
        standard_pr, standard_mcc, standard_pruned = run_evaluation_loop(X, y_noisy, y_true, use_cleanlab=False, al_step_size=al_step_size, iterations=iterations, metric=metric)
        
        # 4. Compile Markdown Table
        markdown_table = (
            "| Iteration | Standard AL (PR-AUC | MCC) | AL-Clean (PR-AUC | MCC) |\n"
            "|-----------|----------------------------|-------------------------|\n"
        )
        for i in range(len(standard_pr)):
            markdown_table += f"| {i+1:<9} | {standard_pr[i]:.4f} | {standard_mcc[i]:.4f} | {clean_pr[i]:.4f} | {clean_mcc[i]:.4f} |\n"
            
        print("\n" + "="*77)
        print("PHASE 4: BENCHMARK RESULTS (PR-AUC & MCC Scores)")
        print("="*77)
        print(markdown_table, end="")
        print("="*77)
        
        # Log the markdown table as an MLflow artifact
        mlflow.log_text(markdown_table, "benchmark_results.md")
        
    print("\nModular Pipeline completed successfully.")

def main():
    run_full_pipeline()

if __name__ == "__main__":
    main()
