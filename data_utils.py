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
    
    # 1. FIXED WEIGHTS (Now 24 values to match 24 hours)
    # Added 0.40 at the end to complete the day cycle
    weights = np.array([
        0.41, 0.42, 0.42, 0.41, 0.42, 0.95, 1.30, 1.56, 1.68, 1.74, 
        1.73, 1.70, 1.71, 1.63, 1.37, 1.35, 1.32, 1.14, 0.99, 0.50, 
        0.44, 0.42, 0.41, 0.40
    ])
    avg_weight = np.mean(weights)
    df['date_only'] = df['fecha'].dt.date

    # Function to reshape stagnant blocks using your profile
    def fix_stagnant(group):
        if (group['consumo_kwh'].max() - group['consumo_kwh'].min()) < tolerance:
            block_avg = group['consumo_kwh'].mean()
            # This applies the weights according to the hour of the day
            group['consumo_kwh'] = [
                block_avg * (weights[h % 24] / avg_weight) 
                for h in group['fecha'].dt.hour.values
            ]
        return group

    # Apply the logic to stagnant days
    df = df.groupby('date_only', group_keys=False).apply(fix_stagnant)

    # 2. PCHIP CURVING & UPSAMPLING
    # To get the "Perfect Curve" look, we need more points than just hours.
    # We will interpolate onto a 15-minute grid.
    
    # Create a numeric representation of time (seconds from start)
    t_numeric = (df['fecha'] - df['fecha'].min()).dt.total_seconds().values
    y_values = df['consumo_kwh'].values
    
    # Build Pchip model (Shape-preserving, prevents overshooting)
    pchip_model = PchipInterpolator(t_numeric, y_values)
    
    # Create a new index with 4x higher resolution (every 15 mins)
    new_t = np.linspace(t_numeric.min(), t_numeric.max(), len(df) * 4)
    new_y = pchip_model(new_t)
    
    # Create the high-resolution dataframe for the chart
    new_dates = [df['fecha'].min() + pd.Timedelta(seconds=s) for s in new_t]
    curved_df = pd.DataFrame({'fecha': new_dates, 'consumo_kwh': new_y})
    
    return curved_df

# --- Streamlit Logic ---
url = "YOUR_DATA_URL"
raw_data = load_github_energy_data(url)

if not raw_data.empty:
    # Processed data now has 15-min granularity and smooth Pchip curves
    curved_data = apply_high_fidelity_filter(raw_data)
    
    st.subheader("High-Fidelity Energy Curve")
    # Setting the index to 'fecha' so Streamlit plots it correctly
    st.line_chart(curved_data.set_index('fecha')['consumo_kwh'])
