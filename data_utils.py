import pandas as pd
import streamlit as st
import io
import requests
import numpy as np

@st.cache_data(show_spinner="Downloading data from GitHub...")
def load_github_energy_data(url: str):
    """
    Fetches energy and temperature data from a GitHub raw URL.
    Handles column renaming and date parsing for consistency.
    """
    try:
        # 1. Read the CSV directly from the URL
        df = pd.read_csv(url, sep=',', skipinitialspace=True)
        
        # 2. Clean column names (removing any hidden spaces)
        df.columns = df.columns.str.strip()
        
        # 3. Standardize Columns
        # We map your specific file columns to friendly names
        rename_map = {
            'Fecha': 'fecha',
            'kWh': 'consumo_kwh',
            'Temp': 'temperatura_c'
        }
        df = df.rename(columns=rename_map)
        
        # 4. Data Type Conversion
        # Parse dates: your file uses DD/MM/YYYY HH:MM:SS
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        
        # Ensure numbers are floats (handles commas if they appear later)
        for col in ['consumo_kwh', 'temperatura_c']:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Cleanup
        # Remove any rows where the date or consumption is missing
        df = df.dropna(subset=['fecha', 'consumo_kwh']).sort_values('fecha')
        
        return df

def apply_high_fidelity_filter(df, threshold_hours=6):
    df['block_id'] = (df['consumo_kwh'].diff() != 0).cumsum()
    daily_profile = np.array([
        0.3, 0.2, 0.2, 0.2, 0.3, 0.5, # 00:00 - 05:00
        0.8, 1.2, 1.5, 1.2, 1.0, 0.9, # 06:00 - 11:00
        0.9, 1.0, 1.1, 1.2, 1.4, 1.8, # 12:00 - 17:00
        2.2, 2.5, 2.3, 1.5, 0.8, 0.5  # 18:00 - 23:00
    ])
    # Normalize it so it averages to 1.0
    daily_profile = daily_profile / daily_profile.mean()

    # 3. Process the blocks
    def reshape_block(group):
        if len(group) >= threshold_hours:
            # Calculate total energy in this flat block
            total_energy = group['consumo_kwh'].sum()
            
            # Map the hours of the group to our daily_profile
            # (using group.fecha.dt.hour to get the right weights)
            weights = daily_profile[group['fecha'].dt.hour.values]
            
            # Re-normalize weights for this specific block length
            weights = weights / weights.sum()
            
            # Redistribute the total energy
            group['consumo_kwh'] = total_energy * weights
            group['is_synthetic'] = True
        else:
            group['is_synthetic'] = False
        return group

    # Apply the reshaper to each block
    df = df.groupby('block_id', group_keys=False).apply(reshape_block)
    
    return df.drop(columns=['block_id'])
    
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
