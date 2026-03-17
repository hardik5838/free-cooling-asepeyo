import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def get_profile_weight(hour_decimal):
    """Returns a weight based on a typical 24h residential curve."""
    hours = np.arange(24)
    weights = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 1.1, 1.4, 1.6, 1.4, 1.2, 1.1,
        1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.8, 1.4, 1.0, 0.7, 0.5
    ])
    f = interp1d(hours, weights, kind='cubic', fill_value="bound", bounds_error=False)
    # Ensure we stay within 0-23 range for the interpolator
    return f(np.clip(hour_decimal, 0, 23))

def apply_high_fidelity_filter(df, tolerance=0.01):
    """
    Identifies flat blocks and redistributes energy based on daily cycles.
    """
    if df.empty:
        return df

    df = df.copy()
    
    # Identify 'stagnant' blocks where consumption doesn't change
    # We use a small tolerance because digital meters rarely hit 'exactly' zero diff
    df['is_stagnant'] = df['consumo_kwh'].diff().abs() <= tolerance
    df['block_id'] = (df['consumo_kwh'].diff().abs() > tolerance).cumsum()

    def redistribute_block(group):
        # We only reshape if the block is 'stagnant' (flat) 
        # and has more than 3 readings (to avoid jitter)
        if group['is_stagnant'].all() and len(group) > 3:
            total_energy = group['consumo_kwh'].sum()
            
            # Calculate weights for each timestamp in the block
            decimal_hours = group['fecha'].dt.hour + group['fecha'].dt.minute / 60.0
            raw_weights = get_profile_weight(decimal_hours)
            
            # Re-normalize: Energy In = Energy Out
            if raw_weights.sum() > 0:
                group['consumo_kwh'] = (raw_weights / raw_weights.sum()) * total_energy
        
        return group

    # Apply redistribution per block
    df = df.groupby('block_id', group_keys=False).apply(redistribute_block)
    
    return df.drop(columns=['block_id', 'is_stagnant'])
