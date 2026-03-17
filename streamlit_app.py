import streamlit as st
from data_utils import load_github_energy_data
from data_utils import apply_high_fidelity_filter

st.set_page_config(page_title="Energy Optimizer", layout="wide")

st.title(" Energy & Weather Dashboard")
st.write("Loading modular data from GitHub for analysis.")

# The Raw URL from GitHub (use the 'Raw' button on GitHub to get this)
GITHUB_URL = "https://raw.githubusercontent.com/hardik5838/free-cooling-asepeyo/refs/heads/main/data/test%20file%20Via%2036%20.csv"

# Using the modular function
raw_data = load_github_energy_data(GITHUB_URL)

if not raw_data.empty:
    st.sidebar.header("Filter Settings")
    
    # TRIGGER: The user chooses to enhance the fidelity
    enhance_fidelity = st.sidebar.toggle("Enhance Data Fidelity", value=True)
    
    if enhance_fidelity:
        # We overwrite the 'data' variable for the WHOLE app
        with st.spinner("Reshaping global energy blocks..."):
            data = apply_high_fidelity_filter(raw_data)
    else:
        data = raw_data

    # --- VISUALIZATION SECTION ---
    
    st.header("Global Consumption Overview")
    # This chart now uses 'data', which is either raw or reshaped globally
    st.line_chart(data.set_index('fecha')['consumo_kwh'], use_container_width=True)

    # --- DETAILED DAY VIEW ---
    st.divider()
    st.subheader("Daily Detail Inspection")
    selected_date = st.date_input("Pick a day to verify the shape", data['fecha'].min())
    
    # Filter the already-processed data for the specific day
    day_view = data[data['fecha'].dt.date == selected_date]
    
    if not day_view.empty:
        st.area_chart(day_view.set_index('fecha')['consumo_kwh'])
    else:
        st.warning("No data for this specific date.")

else:
    st.error("Could not load data. Check your GitHub URL.")
