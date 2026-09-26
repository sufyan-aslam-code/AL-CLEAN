import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="AL-CLEAN Dashboard", layout="wide")
st.title("AL-CLEAN Assessment Dashboard")

st.markdown("### Model Performance vs Data-Cleaning Cycles")

# Sample metrics for demonstration; these should be dynamically loaded in a real integration
iterations = np.arange(1, 6)
standard_pr_auc = [0.65, 0.64, 0.63, 0.61, 0.58]
al_clean_pr_auc = [0.65, 0.70, 0.75, 0.78, 0.82]

df = pd.DataFrame({
    'Iteration': iterations,
    'Standard AL (No Cleanlab)': standard_pr_auc,
    'AL-Clean (With Cleanlab)': al_clean_pr_auc
})

st.line_chart(df.set_index('Iteration'))

st.markdown("### Detailed Metrics Table")
st.dataframe(df)
