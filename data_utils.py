import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from scipy.interpolate import PchipInterpolator

@st.cache_data
def load_github_energy_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep=',', skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={'Fecha': 'fecha', 'kWh': 'consumo_kwh'})
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        return df.dropna(subset=['fecha', 'consumo_kwh']).sort_values('fecha')
    except Exception as e:
        st.error(f"Module Error: {e}")
        return pd.DataFrame()

def apply_high_fidelity_filter(df, stagnation_threshold=3):
    """
    Surgically removes mathematical blocks for clean data analysis.
    No noise addition; pure shape-preserving interpolation.
    """
    if df.empty:
        return df
    
    df = df.copy().reset_index(drop=True)

    # 1. IDENTIFY BLOCK CORES
    # We find where the value stays exactly the same
    is_duplicate = df['consumo_kwh'].diff() == 0
    
    # Identify groups of identical values
    group_id = (~is_duplicate).cumsum()
    counts = df.groupby(group_id)['consumo_kwh'].transform('count')
    
    # We remove rows that are part of a repeating 'block' 
    # but we keep the very first and very last point of the sequence 
    # to serve as accurate boundary conditions for the interpolation.
    is_block_core = (counts >= stagnation_threshold) & is_duplicate
    
    # Remove the 'flat' data points
    df_clean = df[~is_block_core].copy()

    if df_clean.empty:
        return pd.DataFrame(columns=['fecha', 'consumo_kwh'])

    # 2. SHAPE-PRESERVING INTERPOLATION
    # Convert timestamps to numeric seconds
    t_numeric = (df_clean['fecha'] - df_clean['fecha'].min()).dt.total_seconds().values
    y_values = df_clean['consumo_kwh'].values
    
    # Ensure strictly increasing time coordinates
    t_numeric, unique_indices = np.unique(t_numeric, return_index=True)
    y_values = y_values[unique_indices]
    
    # Pchip is preferred over Cubic Spline for analysis because it prevents 
    # 'overshoot'—it won't create artificial peaks or negative consumption.
    pchip_model = PchipInterpolator(t_numeric, y_values)
    
    # Generate the dense timeline (matching your original 15-min intervals)
    # This fills the 'deleted' blocks with a calculated curve
    total_duration = (df['fecha'].max() - df_clean['fecha'].min()).total_seconds()
    t_dense = np.arange(0, total_duration + 900, 900) # 15 mins = 900s
    y_dense = pchip_model(t_dense)
    
    # Rebuild the dataframe
    new_dates = pd.to_datetime(t_dense, unit='s', origin=df_clean['fecha'].min())
    return pd.DataFrame({'fecha': new_dates, 'consumo_kwh': y_dense})

# --- Streamlit Execution ---
url = "YOUR_DATA_URL"
raw_data = load_github_energy_data(url)

if not raw_data.empty:
    # Applying the filter for analysis (No noise, no boxy blocks)
    processed_data = apply_analysis_filter(raw_data, stagnation_threshold=3)
    
    st.subheader("Analytical Consumption Curve (Interpolated)")
    if not processed_data.empty:
        st.line_chart(processed_data.set_index('fecha')['consumo_kwh'])
        
        # Displaying a sample of the cleaned data for verification
        st.write("Cleaned Data Sample:", processed_data.head(10))
    else:
        st.warning("All data filtered out. Check the threshold.")
        
