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
    
    df = df.copy()
    df['date_only'] = df['fecha'].dt.date
    
    # 1. REMOVE STAGNANT BLOCKS
    # Instead of fixing them, we completely drop days where the meter was flat.
    # This identifies days where (Max - Min) is less than your tolerance.
    def is_active(group):
        spread = group['consumo_kwh'].max() - group['consumo_kwh'].min()
        return spread >= tolerance

    # We filter the dataframe to only keep 'Active' days
    df = df.groupby('date_only').filter(is_active)
    
    if df.empty:
        return pd.DataFrame(columns=['fecha', 'consumo_kwh'])

    # 2. PCHIP CURVING & UPSAMPLING
    # We use the remaining "good" hourly data to create the smooth curve.
    # Because the stagnant blocks are gone, Pchip will interpolate 
    # a smooth curve through the resulting gaps.
    
    t_numeric = (df['fecha'] - df['fecha'].min()).dt.total_seconds().values
    y_values = df['consumo_kwh'].values
    
    # Ensure data is sorted and unique for the interpolator
    t_numeric, unique_indices = np.unique(t_numeric, return_index=True)
    y_values = y_values[unique_indices]
    
    # Build Pchip model (Shape-preserving / No overshooting)
    pchip_model = PchipInterpolator(t_numeric, y_values)
    
    # Create 15-minute intervals (4 points per hour) for the organic look
    # We generate a range from the very start to the very end
    t_dense = np.arange(t_numeric.min(), t_numeric.max(), 900) # 900 seconds = 15 mins
    y_dense = pchip_model(t_dense)
    
    # Rebuild high-resolution dataframe
    new_dates = [df['fecha'].min() + pd.Timedelta(seconds=s) for s in t_dense]
    curved_df = pd.DataFrame({'fecha': new_dates, 'consumo_kwh': y_dense})
    
    return curved_df

# --- Streamlit Execution ---
url = "YOUR_DATA_URL"
raw_data = load_github_energy_data(url)

if not raw_data.empty:
    # This version removes blocks and bridges them with smooth curves
    processed_data = apply_high_fidelity_filter(raw_data)
    
    st.subheader("High-Fidelity Filtered Curve")
    if not processed_data.empty:
        st.line_chart(processed_data.set_index('fecha')['consumo_kwh'])
    else:
        st.warning("All data was below the tolerance threshold and was filtered out.")
