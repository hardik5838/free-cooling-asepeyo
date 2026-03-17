import pandas as pd
import requests
import io
import streamlit as st

@st.cache_data
def load_github_energy_data(url):
    try:
        # Fixed: import requests is inside the module now
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

def apply_high_fidelity_filter(df, tolerance=0.01, min_block_size=3):
    """
    Identifies stagnant blocks and removes them (cuts them out).
    """
    if df.empty: return df
    df = df.copy()
    
    # 1. Identify where data stops moving
    df['is_flat'] = df['consumo_kwh'].diff().abs() <= tolerance
    
    # 2. Group these into continuous blocks
    df['block_id'] = (df['consumo_kwh'].diff().abs() > tolerance).cumsum()
    
    # 3. Filter out blocks that are flat and meet the minimum size requirement
    # We keep rows where the block is NOT flat OR the block is very short (jitter)
    def filter_logic(group):
        if group['is_flat'].all() and len(group) >= min_block_size:
            return None # This deletes the block
        return group

    clean_df = df.groupby('block_id', group_keys=False).apply(filter_logic)
    
    return clean_df.drop(columns=['block_id', 'is_flat']).reset_index(drop=True)
