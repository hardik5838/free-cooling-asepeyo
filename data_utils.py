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
    stagnation_threshold: Number of consecutive identical readings 
    required to classify a section as a 'block' and remove it.
    """
    if df.empty:
        return df
    
    df = df.copy().reset_index(drop=True)

    # 1. IDENTIFY AND REMOVE FLAT BLOCKS
    # We look for consecutive rows where the value is identical
    # diff() == 0 means the value is the same as the previous row
    is_duplicate = df['consumo_kwh'].diff() == 0
    
    # We use a rolling count to find blocks longer than our threshold
    # This ensures we don't delete natural 1-hour plateaus, just the long 'boxy' ones
    df['is_block'] = is_duplicate.groupby((~is_duplicate).cumsum()).transform('sum') >= stagnation_threshold
    
    # Drop the blocky sections
    df_clean = df[~df['is_block']].copy()

    if df_clean.empty:
        return pd.DataFrame(columns=['fecha', 'consumo_kwh'])

    # 2. PCHIP INTERPOLATION (The Bridge)
    # We turn the dates into seconds to allow the math to work
    t_numeric = (df_clean['fecha'] - df_clean['fecha'].min()).dt.total_seconds().values
    y_values = df_clean['consumo_kwh'].values
    
    # Remove duplicates in time if any exist
    t_numeric, unique_indices = np.unique(t_numeric, return_index=True)
    y_values = y_values[unique_indices]
    
    # Pchip preserves the 'shape' and prevents the line from going below zero (overshooting)
    pchip_model = PchipInterpolator(t_numeric, y_values)
    
    # Generate a dense 15-minute timeline to replace the missing blocks
    t_dense = np.arange(t_numeric.min(), t_numeric.max(), 900) # 900s = 15m
    y_dense = pchip_model(t_dense)
    
    # Rebuild the high-res dataframe
    new_dates = pd.to_datetime(t_dense, unit='s', origin=df_clean['fecha'].min())
    curved_df = pd.DataFrame({'fecha': new_dates, 'consumo_kwh': y_dense})
    
    return curved_df

# --- Streamlit Execution ---
url = "YOUR_DATA_URL"
raw_data = load_github_energy_data(url)

if not raw_data.empty:
    # We apply the filter to strip out the 'boxy' patterns
    processed_data = apply_high_fidelity_filter(raw_data, stagnation_threshold=3)
    
    st.subheader("Cleaned Curve (Blocks Removed)")
    if not processed_data.empty:
        # Note: We use .line_chart for a clean view
        st.line_chart(processed_data.set_index('fecha')['consumo_kwh'])
    else:
        st.warning("Filter was too aggressive. Try lowering the stagnation_threshold.")
        
