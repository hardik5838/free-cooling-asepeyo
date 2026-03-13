import pandas as pd
import numpy as np
import requests
import io
import streamlit as st

@st.cache_data
def load_github_energy_data(url):
    """Fetches and cleans the CSV from GitHub."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep=',', skipinitialspace=True)
        
        df.columns = df.columns.str.strip()
        # Mapping your specific file columns
        rename_map = {'Fecha': 'fecha', 'kWh': 'consumo_kwh', 'Temp': 'temperatura_c'}
        df = df.rename(columns=rename_map)
        
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        
        return df.dropna(subset=['fecha', 'consumo_kwh']).sort_values('fecha')
    except Exception as e:
        st.error(f"Error Loading: {e}")
        return pd.DataFrame()

def apply_high_fidelity_filter(df, threshold_hours=6):
    """Redistributes flat blocks into a realistic daily shape across the whole file."""
    if df.empty:
        return df
    
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Identify blocks of repeated values
    df['block_id'] = (df['consumo_kwh'].diff() != 0).cumsum()
    
    # Standard Load Profile (Double Peak Shape)
    weights = np.array([
        0.4, 0.3, 0.3, 0.3, 0.4, 0.6, # Night
        1.2, 1.8, 2.0, 1.5, 1.2, 1.1, # Morning Peak
        1.0, 1.1, 1.2, 1.3, 1.5, 2.2, # Mid-day
        2.5, 2.4, 1.8, 1.2, 0.8, 0.5  # Evening Peak
    ])
    
    def reshape_logic(group):
        if len(group) >= threshold_hours:
            total_energy = group['consumo_kwh'].sum()
            group_hours = group['fecha'].dt.hour.values
            specific_weights = weights[group_hours]
            
            # Normalize and distribute
            norm_weights = specific_weights / specific_weights.sum()
            group['consumo_kwh'] = total_energy * norm_weights
        return group

    # Apply to the entire dataset
    df = df.groupby('block_id', group_keys=False).apply(reshape_logic)
    return df.drop(columns=['block_id'])
