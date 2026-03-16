import pandas as pd
import numpy as np
import requests
import io
import streamlit as st

@st.cache_data
def load_github_energy_data(url):
    try:
        response = requests.get(url)
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

def apply_high_fidelity_filter(df, threshold_hours=4, tolerance=5):
    """
    Fuzzy filter that groups values within +/- 1kWh tolerance.
    """
    if df.empty: return df
    df = df.copy()
    
    # Identify 'flat' blocks even if they fluctuate by < 1kWh
    df['block_id'] = (df['consumo_kwh'].diff().abs() > tolerance).cumsum()
    
    # Normalized weights (Avg = 1.0)
    weights = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5
    ])
    weights = weights / weights.mean()

    def reshape_block(group):
        if len(group) >= threshold_hours:
            local_avg = group['consumo_kwh'].mean()
            hours = group['fecha'].dt.hour.values
            # Multiply local avg by the ratio for that hour
            group['consumo_kwh'] = local_avg * weights[hours]
        return group

    return df.groupby('block_id', group_keys=False).apply(reshape_block).drop(columns=['block_id'])
    
