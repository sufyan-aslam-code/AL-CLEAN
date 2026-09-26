import streamlit as st
import pandas as pd
import numpy as np
import requests
import mlflow
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="AL-CLEAN Dashboard", layout="wide")
st.title("AL-CLEAN Assessment Dashboard")

st.sidebar.header("Pipeline Configuration")
noise_level = st.sidebar.slider("Synthetic Label Noise", min_value=0.05, max_value=0.30, value=0.15, step=0.01)
iterations = st.sidebar.number_input("Active Learning Iterations", min_value=1, max_value=10, value=5)
sampling_metric = st.sidebar.selectbox("Sampling Metric", ["entropy", "margin"])

st.sidebar.markdown("---")
display_metric = st.sidebar.radio("Select Evaluation Metric", ["PR-AUC", "MCC", "Budget Savings"])

if st.sidebar.button("Run Pipeline"):
    if "run_id" in st.session_state:
        del st.session_state["run_id"]
    with st.spinner("Executing Data-Cleaning Cycles..."):
        payload = {
            "noise_level": noise_level,
            "iterations": int(iterations),
            "sampling_metric": sampling_metric
        }
        try:
            response = requests.post("http://localhost:8000/run-pipeline", json=payload)
            if response.status_code == 200:
                st.session_state.run_id = response.json().get("run_id")
                st.sidebar.success("Pipeline Execution Completed!")
                st.rerun()
            else:
                st.sidebar.error(f"Failed to run pipeline: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Connection error: {e}")

st.markdown("### Model Performance vs Data-Cleaning Cycles")

# Try to fetch from MLflow if tracking URI is set or reachable
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    if "run_id" not in st.session_state:
        runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)
        if runs:
            st.session_state.run_id = runs[0].info.run_id
            
    if "run_id" in st.session_state:
        run_id = st.session_state.run_id
        
        std_pr = client.get_metric_history(run_id, "Standard_AL_PR_AUC")
        cln_pr = client.get_metric_history(run_id, "AL-Clean_PR_AUC")
        std_mcc = client.get_metric_history(run_id, "Standard_AL_MCC")
        cln_mcc = client.get_metric_history(run_id, "AL-Clean_MCC")
        
        try:
            std_budget = client.get_metric_history(run_id, "Standard_AL_Budget_Savings")
            cln_budget = client.get_metric_history(run_id, "AL-Clean_Budget_Savings")
        except:
            std_budget = []
            cln_budget = []
        
        if std_pr and cln_pr and std_mcc and cln_mcc:
            iters = [m.step for m in std_pr]
            
            df_dict = {
                'Iteration': iters,
                'Standard AL PR-AUC': [m.value for m in std_pr],
                'AL-Clean PR-AUC': [m.value for m in cln_pr],
                'Standard AL MCC': [m.value for m in std_mcc],
                'AL-Clean MCC': [m.value for m in cln_mcc]
            }
            
            if cln_budget:
                # Add Budget Savings if available
                # Assuming std_budget and cln_budget lists match 'iters'
                df_dict['Standard AL Budget Savings'] = [m.value for m in std_budget] if std_budget else [0] * len(iters)
                df_dict['AL-Clean Budget Savings'] = [m.value for m in cln_budget]
                
            df = pd.DataFrame(df_dict)
        else:
            st.info("Pipeline is still executing in the background. Refresh soon to see data...")
            df = pd.DataFrame()
    else:
        st.info("No previous runs found. Please configure the parameters and click 'Run Pipeline' to generate data.")
        df = pd.DataFrame()
except Exception as e:
    st.error(f"MLflow fetch failed: {e}")
    df = pd.DataFrame()

if not df.empty:
    if (df[f'Standard AL {display_metric}'] == 0).all():
        st.warning(f"Warning: All {display_metric} metrics returned 0.0. This may indicate a class imbalance issue where all minority classes were dropped during subsampling or cross-validation.")
    
    if display_metric == "Budget Savings" and 'AL-Clean Budget Savings' not in df.columns:
        st.warning("Budget Savings data is not available for this run.")
        plot_df = pd.DataFrame()
    else:
        # Dynamically plot the selected metric
        plot_df = df[['Iteration', f'Standard AL {display_metric}', f'AL-Clean {display_metric}']].set_index('Iteration')
        st.line_chart(plot_df)
    
    st.markdown("### Detailed Metrics Table")
    st.dataframe(df)
