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
        response.raise_for_status() 
        df = pd.read_csv(io.StringIO(response.text), sep=',', skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # Mapping: the file has 'kWh' and 'Fecha'
        df = df.rename(columns={'Fecha': 'fecha', 'kWh': 'consumo_kwh'})
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        
        return df.dropna(subset=['fecha', 'consumo_kwh']).sort_values('fecha')
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

def get_daily_profile(decimal_hours):
    """Calculates weights for a 24h cycle using cubic interpolation."""
    hours_idx = np.arange(24)
    weights = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5
    ])
    f_interp = interp1d(hours_idx, weights, kind='cubic', fill_value="wrap", bounds_error=False)
    # Use modulo 24 to handle multi-day blocks seamlessly
    return f_interp(decimal_hours % 24)

def apply_high_fidelity_filter(df, tolerance=0.01):
    if df.empty: return df
    df = df.copy()

    # Identify blocks of stagnant (flat) readings
    # A block ends when the difference between readings exceeds the tolerance
    df['is_flat'] = df['consumo_kwh'].diff().abs() <= tolerance
    df['block_id'] = (df['consumo_kwh'].diff().abs() > tolerance).cumsum()

    def reshape_block(group):
        # We only apply the curve if the block is flat and has enough data points
        if group['is_flat'].all() and len(group) > 2:
            total_energy = group['consumo_kwh'].sum()
            
            # Use precise time (decimal hours) to get a smooth weight
            decimal_hours = group['fecha'].dt.hour + group['fecha'].dt.minute / 60.0
            smooth_weights = get_daily_profile(decimal_hours)
            
            # Redistribute total energy based on weights (Energy Conservation)
            if smooth_weights.sum() > 0:
                group['consumo_kwh'] = (smooth_weights / smooth_weights.sum()) * total_energy
        return group

    return df.groupby('block_id', group_keys=False).apply(reshape_block).drop(columns=['block_id', 'is_flat'])
