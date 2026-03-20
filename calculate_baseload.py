import pandas as pd
import numpy as np

def calculate_baseload(df):
    """
    BLOCK 4: BASELOAD CALCULATION
    Identifies the steady-state parasitic load using the bottom 5th percentile.
    """
    if df.empty:
        return 0
    
    # 1. Remove zeros (ignore blackouts/sensor errors)
    non_zero_data = df[df['consumo_kwh'] > 0.1].copy()
    
    if non_zero_data.empty:
        return 0

    # 2. Use the 5th percentile to find the 'floor' of consumption
    threshold = non_zero_data['consumo_kwh'].quantile(0.05)
    
    # 3. Median of these low points is our Baseload
    baseload = non_zero_data[non_zero_data['consumo_kwh'] <= threshold]['consumo_kwh'].median()
    
    return round(float(baseload), 2)
