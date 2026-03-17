import pandas as pd
import numpy as np
import requests  # <--- CRITICAL: This fixes the "name 'requests' is not defined" error
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

        df = df.rename(columns={'Fecha': 'fecha', 'kWh': 'consumo_kwh'})
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        
        return df.dropna(subset=['fecha', 'consumo_kwh']).sort_values('fecha')
    except Exception as e:
        st.error(f"Module Error: {e}")
        return pd.DataFrame()

def apply_high_fidelity_filter(df, tolerance=0.02):
    """Reshapes flat blocks without butchering the energy totals."""
    if df.empty: return df
    df = df.copy()
    
    # Identify blocks where data is 'stuck' (flat)
    df['is_flat'] = df['consumo_kwh'].diff().abs() <= tolerance
    df['block_id'] = (df['consumo_kwh'].diff().abs() > tolerance).cumsum()

    # Residential 24h weights
    hours_idx = np.arange(24)
    weights = np.array([0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
                        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5])
    f_interp = interp1d(hours_idx, weights, kind='cubic', fill_value="wrap", bounds_error=False)

    def reshape_block(group):
        # Only reshape if the block is flat and spans a significant time (e.g. > 3 hours)
        if group['is_flat'].all() and len(group) > 3:
            total_energy = group['consumo_kwh'].sum()
            # Calculate weight for every specific timestamp in the block
            decimal_hours = group['fecha'].dt.hour + group['fecha'].dt.minute / 60.0
            smooth_weights = f_interp(decimal_hours % 24)
            
            # Re-normalize: ensure the sum of new values equals the original total energy
            if smooth_weights.sum() > 0:
                group['consumo_kwh'] = (smooth_weights / smooth_weights.sum()) * total_energy
        return group

    return df.groupby('block_id', group_keys=False).apply(reshape_block).drop(columns=['block_id', 'is_flat'])
