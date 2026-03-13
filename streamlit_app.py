import streamlit as st
from data_utils import load_github_energy_data

st.set_page_config(page_title="Energy Analysis", layout="wide")

st.title("⚡ Energy & Weather Dashboard")
st.write("Loading modular data from GitHub for analysis.")

# The Raw URL from GitHub (use the 'Raw' button on GitHub to get this)
GITHUB_URL = "https://raw.githubusercontent.com/your_user/your_repo/main/test_file_Via_36.csv"

# Using the modular function
data = load_github_energy_data(GITHUB_URL)

if not data.empty:
    st.success("Data loaded successfully!")
    
    # Quick metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(data))
    col2.metric("Avg Temp", f"{data['temperatura_c'].mean():.2f} °C")
    col3.metric("Max Consumption", f"{data['consumo_kwh'].max():.2f} kWh")
    
    # Preview
    st.dataframe(data.head(10), use_container_width=True)
    
    # Quick Plot
    st.line_chart(data.set_index('fecha')[['consumo_kwh']])
else:
    st.warning("Waiting for data... check the URL or file format.")
