import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from scipy.interpolate import interp1d

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
    
    # 1. FIX STAGNANT BLOCKS (Keep your weight logic)
    weights = np.array([0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
                        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5])
    avg_weight = np.mean(weights)
    df['date_only'] = df['fecha'].dt.date

    def fix_stagnant(group):
        if (group['consumo_kwh'].max() - group['consumo_kwh'].min()) < tolerance:
            block_avg = group['consumo_kwh'].mean()
            hours = group['fecha'].dt.hour.values
            group['consumo_kwh'] = [block_avg * (weights[h % 24] / avg_weight) for h in hours]
        return group

    df = df.groupby('date_only', group_keys=False).apply(fix_stagnant)

    # 2. THE CUBIC SPLINE (This is where the 'Smooth Curve' happens)
    # We treat every single hour as a 'knot' to preserve granularity
    x = np.arange(len(df))
    y = df['consumo_kwh'].values
    
    # Kind='cubic' creates that smooth, non-linear flow between points
    spline_model = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    
    # We create a 'dense' version (e.g., 4x more points) to make the line look smooth
    x_dense = np.linspace(0, len(df) - 1, len(df) * 4)
    y_dense = spline_model(x_dense)
    
    # Because we need to return the same shape to your dataframe, 
    # we evaluate the spline back at the original hourly marks.
    df['consumo_kwh'] = spline_model(x)
    
    return df.drop(columns=['date_only'])
