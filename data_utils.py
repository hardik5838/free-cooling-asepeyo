import pandas as pd
import numpy as np

def apply_high_fidelity_filter(df, threshold_hours=3, tolerance=1.0):
    """
    Fuzzy High-Fidelity Filter:
    - tolerance: ignores changes smaller than this (e.g., 1 kWh) to find 'flat' blocks.
    - Uses local block average to scale the synthetic curve.
    """
    if df.empty: return df
    df = df.copy()
    
    # 1. FUZZY DETECTION: Identify where the value 'really' changes
    # We create a mask where the jump is greater than our tolerance (1 kWh)
    df['significant_change'] = df['consumo_kwh'].diff().abs() > tolerance
    df['block_id'] = df['significant_change'].cumsum()
    
    # 2. THE HOURLY RATIO MAP (Standardized for a typical building)
    # These represent 'Percentage of the Daily Average'
    # e.g., 0.5 = 50% of avg, 1.8 = 180% of avg.
    ratios = np.array([
        0.55, 0.45, 0.40, 0.40, 0.45, 0.65, # 00-05: Deep night (low)
        1.10, 1.45, 1.70, 1.50, 1.30, 1.20, # 06-11: Morning ramp up
        1.15, 1.20, 1.30, 1.40, 1.55, 1.85, # 12-17: Afternoon steady/rise
        2.10, 1.90, 1.50, 1.10, 0.80, 0.65  # 18-23: Evening peak & drop
    ])
    
    # Normalize ratios so the average is exactly 1.0
    # This ensures that (Local Average * Ratio) preserves the total energy
    ratios = ratios / ratios.mean()

    def reshape_block(group):
        # Only process blocks that look 'stuck' or 'averaged'
        if len(group) >= threshold_hours:
            # Calculate the local average of this specific 5h or 24h block
            local_avg = group['consumo_kwh'].mean()
            
            # Map the specific hours of this block to our ratio map
            hours = group['fecha'].dt.hour.values
            block_ratios = ratios[hours]
            
            # RE-SCALE: Each cell becomes (Local Average * Its Hour Ratio)
            # We add a tiny bit of random noise (2%) to make the curve 'breakable'
            noise = np.random.uniform(0.98, 1.02, size=len(group))
            group['consumo_kwh'] = local_avg * block_ratios * noise
            
        return group

    # Apply to the whole file
    df = df.groupby('block_id', group_keys=False).apply(reshape_block)
    
    # Cleanup temporary columns
    return df.drop(columns=['significant_change', 'block_id'])
