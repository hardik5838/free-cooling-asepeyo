import streamlit as st
import requests
import pandas as pd
from data_utils import load_github_energy_data, apply_high_fidelity_filter
from feature import build_oasis_features

st.set_page_config(page_title="Oasis Energy Optimizer", layout="wide")

st.title("Oasis Project: Energy & Weather Dashboard")
st.write("Transitioning from Digital Shadows to Active Digital Twins.")

# 1. LOAD DATA
GITHUB_URL = "https://raw.githubusercontent.com/hardik5838/free-cooling-asepeyo/refs/heads/main/data/test%20file%20Via%2036%20.csv"
raw_data = load_github_energy_data(GITHUB_URL)

# --- GUARD CLAUSE: Exit early if data is missing ---
if raw_data.empty:
    st.error("Could not load data. Check your GitHub URL or internet connection.")
    st.stop() # This prevents the rest of the code from running

# --- If we are here, data exists! ---

# 2. CONTROL PANEL
st.sidebar.header("Control Panel")
enhance_fidelity = st.sidebar.toggle("Step 1: Enhance Data Fidelity (Block 1)", value=True)

if enhance_fidelity:
    with st.spinner("Reshaping global energy blocks..."):
        data = apply_high_fidelity_filter(raw_data)
else:
    data = raw_data

# 3. FEATURE ENGINEERING (BLOCK 3)
st.sidebar.divider()
run_features = st.sidebar.toggle("Step 2: Generate AI Features (Block 3)", value=True)

# Initialize model_data as empty so the app doesn't crash if toggle is off
model_data = pd.DataFrame() 

if run_features:
    with st.spinner("Generating temporal and cyclical features..."):
        model_data = build_oasis_features(data)

# --- 4. VISUALIZATION SECTION ---
st.header("Global Consumption Overview")

with st.expander("View Global Trend (Resampled)", expanded=True):
    # SPEED FIX: Resample to Daily for the big chart
    global_chart_data = data.set_index('fecha')['consumo_kwh'].resample('D').mean()
    st.line_chart(global_chart_data, use_container_width=True)
    st.caption("Chart shows daily averages to maintain performance.")

# --- 5. FEATURE PREVIEW SECTION ---
if run_features and not model_data.empty:
    st.divider()
    st.subheader("The Feature Matrix (Oasis Inputs)")
    st.write("These columns (Lags, Hour, Weekend) are what the XGBoost will use to predict.")
    
    # Show a snippet of the table
    st.dataframe(model_data.tail(10))
    
    # PROOF OF CONCEPT: Compare Actual vs. 7-Day Lag
    st.write("Checking 'Memory' Alignment: Actual vs. Last Week")
    # Tail(500) gives us a clear look at the last few days
    comparison_plot = model_data.tail(500).set_index('fecha')[['consumo_kwh', 'lag_7d']]
    st.line_chart(comparison_plot)
