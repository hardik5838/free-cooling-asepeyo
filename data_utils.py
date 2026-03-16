import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from scipy.interpolate import interp1d

@st.cache_data
def load_github_energy_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Check if URL is actually accessible
        df = pd.read_csv(io.StringIO(response.text), sep=',', skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # Mapping: the file has 'kWh' and 'Fecha'
        df = df.rename(columns={'Fecha': 'fecha', 'kWh': 'consumo_kwh'})
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        
        return df.dropna(subset=['fecha', 'consumo_kwh']).sort_values('fecha')
    except Exception as e:
        st.error(f"Module Error: {e}")
        return pd.DataFrame()

def apply_high_fidelity_filter(df, threshold_hours=24, tolerance=0.1):
    if df.empty: 
        return df
    
    df = df.copy()
    
    # 1. Identify blocks of stagnant data
    df['block_id'] = (df['consumo_kwh'].diff().abs() > tolerance).cumsum()
    
    # 2. Define the 24-hour profile
    hours_idx = np.arange(24)
    weights = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5
    ])
    
    # Cubic spline for smooth curves
    f_interp = interp1d(hours_idx, weights, kind='cubic', fill_value="extrapolate")

    def reshape_block(group):
        if len(group) >= threshold_hours:
            total_energy = group['consumo_kwh'].sum()
            # Use decimal hours for a smooth curve
            precise_hours = group['fecha'].dt.hour + group['fecha'].dt.minute / 60.0
            smooth_weights = f_interp(precise_hours)
            
            # Re-normalize weights to ensure energy conservation
            group['consumo_kwh'] = (smooth_weights / smooth_weights.sum()) * total_energy
        return group

    return df.groupby('block_id', group_keys=False).apply(reshape_block).drop(columns=['block_id'])
