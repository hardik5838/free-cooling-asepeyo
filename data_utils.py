import pandas as pd
import streamlit as st
import io
import requests

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
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
