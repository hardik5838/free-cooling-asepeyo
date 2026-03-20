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

# --- 7. DETAILED LOAD BREAKDOWN (APPENDAGE) ---
st.divider()
st.header("🔍 Daily Energy 'Slice' Inspection")
st.write("Isolating the Variable Load (HVAC/Lighting) from the Static Baseload.")

# 1. Date Selection
selected_date = st.date_input("Select a day to inspect", data['fecha'].max())

# 2. Filter data for that day
day_data = data[data['fecha'].dt.date == selected_date].copy()

if not day_data.empty:
    # 3. Disaggregation Logic (Simplified for Visualization)
    day_data = day_data.set_index('fecha')
    day_data['Base_Load'] = baseload_val
    # Variable Load is everything above the floor
    day_data['Variable_Load'] = (day_data['consumo_kwh'] - baseload_val).clip(lower=0)
    
    # 4. Visualization: Stacked Area Chart
    # This clearly shows the 'Parasitic' floor vs the 'Active' usage
    st.subheader(f"Energy Distribution for {selected_date}")
    st.area_chart(day_data[['Base_Load', 'Variable_Load']], color=["#7f8c8d", "#2ecc71"])

    # 5. Efficiency Insights (Financial Tooling)
    var_total = day_data['Variable_Load'].sum()
    base_total = day_data['Base_Load'].sum()
    total_day = day_data['consumo_kwh'].sum()
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Variable Energy (HVAC/Light)", f"{var_total:.1f} kWh")
        st.caption("Energy influenced by occupancy/weather.")
    with col_b:
        efficiency_ratio = (var_total / total_day) * 100
        st.metric("Efficiency Ratio", f"{efficiency_ratio:.1%}")
        st.caption("Percentage of energy used for active tasks.")
    with col_c:
        # Optimization Goal: 15% reduction in Variable Load
        potential_saving = var_total * 0.15 * 0.15 # 15% reduction @ €0.15/kWh
        st.metric("Oasis Optimization Target", f"€{potential_saving:.2f}", delta="-15%")
        st.caption("Daily saving potential via Oasis Control.")
else:
    st.warning("No high-fidelity data available for the selected date.")
