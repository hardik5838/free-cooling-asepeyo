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

def apply_high_fidelity_filter(df, stagnation_threshold=3, noise_level=1.5):
    """
    stagnation_threshold: Number of repeated identical values to trigger deletion.
    noise_level: Amount of 'jitter' to add to the smooth line to make it look real.
    """
    if df.empty:
        return df
    
    df = df.copy().reset_index(drop=True)

    # 1. IDENTIFY AND DELETE BLOCKS
    # We find where the value stays exactly the same
    is_duplicate = df['consumo_kwh'].diff() == 0
    
    # Identify groups of identical values
    group_id = (~is_duplicate).cumsum()
    counts = df.groupby(group_id)['consumo_kwh'].transform('count')
    
    # We remove the middle of the blocks, but keep the first and last points 
    # of the transition to act as anchors for the curve.
    is_block = (counts >= stagnation_threshold) & is_duplicate
    
    # Filter: Keep only data that isn't a 'stuck' repeated value
    df_clean = df[~is_block].copy()

    if df_clean.empty:
        return pd.DataFrame(columns=['fecha', 'consumo_kwh'])

    # 2. PCHIP INTERPOLATION (The Bridge)
    t_numeric = (df_clean['fecha'] - df_clean['fecha'].min()).dt.total_seconds().values
    y_values = df_clean['consumo_kwh'].values
    
    # Ensure time is unique for the math model
    t_numeric, unique_indices = np.unique(t_numeric, return_index=True)
    y_values = y_values[unique_indices]
    
    # Pchip creates an organic curve without 'overshooting' (going negative)
    pchip_model = PchipInterpolator(t_numeric, y_values)
    
    # Create a dense 15-minute timeline for the whole range
    total_seconds = (df['fecha'].max() - df_clean['fecha'].min()).total_seconds()
    t_dense = np.arange(0, total_seconds, 900) # 900s = 15m
    y_dense = pchip_model(t_dense)
    
    # 3. ADD SYNTHETIC TEXTURE (Noise)
    # The later months have jitter; the earlier months are too smooth after Pchip.
    # We add a small random variance so the graph looks consistent.
    jitter = np.random.normal(0, noise_level, len(y_dense))
    y_dense_noisy = y_dense + jitter
    
    # Rebuild the high-res dataframe
    new_dates = pd.to_datetime(t_dense, unit='s', origin=df_clean['fecha'].min())
    curved_df = pd.DataFrame({'fecha': new_dates, 'consumo_kwh': y_dense_noisy})
    
    return curved_df

# --- Streamlit Execution ---
# Replace with your actual CSV URL
url = "YOUR_DATA_URL"
raw_data = load_github_energy_data(url)

if not raw_data.empty:
    # We use a threshold of 3. For your data (108.9 repeating 20+ times), 
    # this will effectively wipe the plateaus and bridge them.
    processed_data = apply_high_fidelity_filter(raw_data, stagnation_threshold=3, noise_level=2.0)
    
    st.subheader("High-Fidelity Consumption Curve")
    if not processed_data.empty:
        st.line_chart(processed_data.set_index('fecha')['consumo_kwh'])
    else:
        st.warning("All data was detected as blocky. Try increasing the threshold.")
