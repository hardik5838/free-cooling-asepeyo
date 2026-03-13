# --- data_utils.py ---
import pandas as pd
import numpy as np
import requests
import io
import streamlit as st

def load_github_energy_data(url):
    """Fetches and cleans the CSV from GitHub."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Check if the URL is valid
        
        # Read the raw text from the GitHub response
        df = pd.read_csv(io.StringIO(response.text), sep=',', skipinitialspace=True)
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        rename_map = {'Fecha': 'fecha', 'kWh': 'consumo_kwh', 'Temp': 'temperatura_c'}
        df = df.rename(columns=rename_map)
        
        # Convert types
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        
        return df.dropna(subset=['fecha', 'consumo_kwh']).sort_values('fecha')
    except Exception as e:
        st.error(f"Error in data_utils.load_github_energy_data: {e}")
        return pd.DataFrame()

def apply_high_fidelity_filter(df, threshold_hours=6):
    """Redistributes flat blocks into a realistic daily shape."""
    if df.empty:
        return df
        
    # Detect blocks of repeated values
    df = df.copy()
    df['block_id'] = (df['consumo_kwh'].diff() != 0).cumsum()
    
    # Standard Load Profile (Weibull-inspired double peak)
    # 24 values representing weights for each hour (00:00 to 23:00)
    weights = np.array([
        0.4, 0.3, 0.3, 0.3, 0.4, 0.6, # 00-05 Night
        1.2, 1.8, 2.0, 1.5, 1.2, 1.1, # 06-11 Morning Peak
        1.0, 1.1, 1.2, 1.3, 1.5, 2.2, # 12-17 Afternoon
        2.5, 2.4, 1.8, 1.2, 0.8, 0.5  # 18-23 Evening Peak
    ])
    
    def reshape_logic(group):
        if len(group) >= threshold_hours:
            total_energy = group['consumo_kwh'].sum()
            # Map weights to the actual hours of these rows
            group_hours = group['fecha'].dt.hour.values
            specific_weights = weights[group_hours]
            
            # Normalize so the sum of weights = 1, then multiply by total energy
            norm_weights = specific_weights / specific_weights.sum()
            group['consumo_kwh'] = total_energy * norm_weights
        return group

    new_df = df.groupby('block_id', group_keys=False).apply(reshape_logic)
    return new_df.drop(columns=['block_id'])
