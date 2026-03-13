import streamlit as st
from data_utils import load_github_energy_data, apply_high_fidelity_filter

st.set_page_config(page_title="Energy Analysis", layout="wide")

st.title(" Energy & Weather Dashboard")
st.write("Loading modular data from GitHub for analysis.")

# The Raw URL from GitHub (use the 'Raw' button on GitHub to get this)
GITHUB_URL = "https://raw.githubusercontent.com/hardik5838/free-cooling-asepeyo/refs/heads/main/data/test%20file%20Via%2036%20.csv"

# Using the modular function
data = load_github_energy_data(GITHUB_URL)

if not data.empty:
    st.sidebar.header("Filter Settings")
    use_filter = st.sidebar.checkbox("Apply High-Fidelity Filter", value=True)
    
    if use_filter:
        processed_data = apply_high_fidelity_filter(data.copy())
        st.info("💡 High-Fidelity Filter active: Flat periods have been redistributed using a standard load profile.")
    else:
        processed_data = data

    # Visualization
    st.subheader("Energy Profile Comparison")
    
    # Select a specific date to see the 'Shape' clearly
    selected_date = st.date_input("Focus on a specific day:", processed_data['fecha'].min())
    day_data = processed_data[processed_data['fecha'].dt.date == selected_date]
    
    st.line_chart(day_data.set_index('fecha')[['consumo_kwh']])



if not data.empty:
    st.success("Data loaded successfully!")
    
    # Quick metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(data))
    col2.metric("Avg Temp", f"{data['temperatura_c'].mean():.2f} °C")
    col3.metric("Max Consumption", f"{data['consumo_kwh'].max():.2f} kWh")
    
    # Quick Plot
    st.line_chart(data.set_index('fecha')[['consumo_kwh']])
else:
    st.warning("Waiting for data... check the URL or file format.")
