import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from scipy.signal import savgol_filter

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
    
    df = df.copy()
    
    # 1. Define the "Normal" 24-hour Profile Weights
    # These weights ensure we keep the "Daily Highs and Lows"
    weights = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5
    ])
    avg_weight = np.mean(weights)
    
    # 2. Identify and "Curve" the Stagnant Blocks
    df['date_only'] = df['fecha'].dt.date
    
    def fix_block(group):
        # We check for a full day (24 hours) or stagnant data
        spread = group['consumo_kwh'].max() - group['consumo_kwh'].min()
        
        if spread < tolerance:
            # This is a 'flat' block. We reshape it using the weights.
            block_avg = group['consumo_kwh'].mean()
            hours = group['fecha'].dt.hour.values
            
            new_values = []
            for h in hours:
                h_idx = int(h) % 24
                corrected_val = block_avg * (weights[h_idx] / avg_weight)
                new_values.append(corrected_val)
            
            group['consumo_kwh'] = new_values
        return group

    # Apply the profile to flat days
    df = df.groupby('date_only', group_keys=False).apply(fix_block)

    # 3. Apply High-Granularity Smoothing (The "Cubic Spline" effect)
    # Savitzky-Golay with a small window (e.g., 5 or 7 hours)
    # This turns the "steps" between hours into smooth curves 
    # WITHOUT losing the daily peaks.
    window_size = 7  # Must be odd. Smaller = more granularity, Larger = smoother.
    if len(df) > window_size:
        df['consumo_kwh'] = savgol_filter(df['consumo_kwh'], 
                                         window_length=window_size, 
                                         polyorder=3) # polyorder 3 = Cubic behavior

    return df.drop(columns=['date_only'])

# --- Streamlit Usage ---
url = "YOUR_DATA_URL"
raw_data = load_github_energy_data(url)

if not raw_data.empty:
    processed_data = apply_high_fidelity_filter(raw_data)
    
    st.subheader("High-Granularity Curved Data")
    # This will now show the daily peaks/valleys but as a smooth curve
    st.line_chart(processed_data.set_index('fecha')['consumo_kwh'])
