import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from scipy.interpolate import interp1d # Efficient cubic spline tool

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

def apply_high_fidelity_filter(df):
    """
    Transforms 'blocks of average' into a smooth continuous curve 
    using Cubic Spline interpolation.
    """
    if df.empty or len(df) < 4:  # Spline needs at least 4 points for 'cubic'
        return df
    
    df = df.copy()
    
    # 1. Create the "Blocks" (Daily Averages)
    # This finds the mean consumption for every day in the dataset
    daily = df.groupby(df['fecha'].dt.date)['consumo_kwh'].mean().reset_index()
    daily['fecha'] = pd.to_datetime(daily['fecha'])
    
    # 2. Define the "Knots" for the Spline
    # We place the daily average value at the middle of each day (12:00 PM)
    daily['knot_time'] = daily['fecha'] + pd.Timedelta(hours=12)
    
    # 3. Convert Time to Numeric for Scipy
    # Interpolation requires numeric X values (seconds since start)
    start_time = df['fecha'].min()
    daily_x = (daily['knot_time'] - start_time).dt.total_seconds()
    daily_y = daily['consumo_kwh']
    
    # 4. Build the Cubic Spline Model
    # 'cubic' creates the smooth curve that reshapes the blocky data
    spline_model = interp1d(daily_x, daily_y, kind='cubic', fill_value="extrapolate")
    
    # 5. Reshape the original hourly data
    # We evaluate the spline at every actual hourly timestamp in your data
    df_x = (df['fecha'] - start_time).dt.total_seconds()
    df['consumo_kwh'] = spline_model(df_x)
    
    # Safety: Energy consumption cannot be negative
    df['consumo_kwh'] = df['consumo_kwh'].clip(lower=0)
    
    return df

# --- Streamlit Execution Example ---
st.title("Energy Consumption Curve Generator")
url = "YOUR_GITHUB_CSV_URL" # Replace with your actual URL

data = load_github_energy_data(url)
if not data.empty:
    st.subheader("Original Data (with blocks)")
    st.line_chart(data.set_index('fecha')['consumo_kwh'])

    # Apply the Cubic Spline Transformation
    curved_data = apply_high_fidelity_filter(data)

    st.subheader("Reshaped Data (Cubic Spline)")
    st.line_chart(curved_data.set_index('fecha')['consumo_kwh'])
