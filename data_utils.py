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
    if df.empty:
        return df
    
    df = df.copy()
    
    # 1. Define the "Normal" 24-hour Profile Weights
    # These represent the typical shape of your energy curve
    weights = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5
    ])
    avg_weight = np.mean(weights)
    
    # 2. Identify "Weird" Blocks (where variance is below tolerance over 24h)
    # We group by date to check 24-hour chunks
    df['date_only'] = df['fecha'].dt.date
    
    def fix_block(group):
        # Only fix if the block is actually 24 entries (1 day) 
        # and the spread (max-min) is less than your tolerance
        if len(group) == 24:
            spread = group['consumo_kwh'].max() - group['consumo_kwh'].min()
            
            if spread < tolerance:
                # Calculate the 24h average of this weird block
                block_avg = group['consumo_kwh'].mean()
                
                # Apply the correlation: X = Block_Avg * (Weight_h / Avg_Weight)
                # This ensures the total energy for the day remains the same
                hours = group['fecha'].dt.hour.values
                new_values = []
                for h in hours:
                    corrected_val = block_avg * (weights[h] / avg_weight)
                    new_values.append(corrected_val)
                
                group['consumo_kwh'] = new_values
        return group

    # Apply the logic and cleanup
    df = df.groupby('date_only', group_keys=False).apply(fix_block)
    return df.drop(columns=['date_only'])


