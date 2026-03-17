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

def apply_high_fidelity_filter(df, tolerance=1.0):
    if df.empty:
        return df
    
    df = df.copy().reset_index(drop=True)
    
    # 1. FIXED WEIGHTS (Now exactly 24 values to prevent IndexError)
    weights = np.array([
        0.41, 0.42, 0.42, 0.41, 0.42, 0.95, 1.30, 1.56, 1.68, 1.74, 
        1.73, 1.70, 1.71, 1.63, 1.37, 1.35, 1.32, 1.14, 0.99, 0.50, 
        0.44, 0.42, 0.41, 0.40  # Added 24th value
    ])
    avg_weight = np.mean(weights)
    df['date_only'] = df['fecha'].dt.date

    def fix_stagnant(group):
        # Only reshape if the data is "flat" (stagnant meter)
        if (group['consumo_kwh'].max() - group['consumo_kwh'].min()) < tolerance:
            block_avg = group['consumo_kwh'].mean()
            # Safety: use % 24 to ensure we never exceed the array length
            group['consumo_kwh'] = [
                block_avg * (weights[int(h) % 24] / avg_weight) 
                for h in group['fecha'].dt.hour.values
            ]
        return group

    # Process daily blocks
    df = df.groupby('date_only', group_keys=False).apply(fix_stagnant)

    # 2. PCHIP CURVING & UPSAMPLING (For the organic look)
    # We turn the 1-hour data into 15-minute data to make it look curved
    t_numeric = (df['fecha'] - df['fecha'].min()).dt.total_seconds().values
    y_values = df['consumo_kwh'].values
    
    # Shape-preserving spline (No overshooting)
    pchip_model = PchipInterpolator(t_numeric, y_values)
    
    # Create 4x more points (15 min intervals)
    t_dense = np.linspace(t_numeric.min(), t_numeric.max(), len(df) * 4)
    y_dense = pchip_model(t_dense)
    
    # Rebuild high-res dataframe
    new_dates = [df['fecha'].min() + pd.Timedelta(seconds=s) for s in t_dense]
    curved_df = pd.DataFrame({'fecha': new_dates, 'consumo_kwh': y_dense})
    
    return curved_df

# --- Streamlit Usage ---
# data = apply_high_fidelity_filter(raw_data)
# st.line_chart(data.set_index('fecha'))
