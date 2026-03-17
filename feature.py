import pandas as pd
import numpy as np

def build_oasis_features(df, temp_df=None):
    """
    Input: Cleaned dataframe from Block 1.
    Output: Dataframe with lags, time features, and climate interactions.
    """
    # Ensure we don't overwrite the original data
    feat_df = df.copy().sort_values('fecha')

    # 1. TEMPORAL FEATURES (The 'Habit' of the building)
    feat_df['hour'] = feat_df['fecha'].dt.hour
    feat_df['day_of_week'] = feat_df['fecha'].dt.dayofweek
    feat_df['month'] = feat_df['fecha'].dt.month
    feat_df['is_weekend'] = feat_df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. OCCUPANCY PROXY (Binary Logic)
    # Most office buildings ramp up at 7am and down at 7pm
    feat_df['is_business_hours'] = feat_df['hour'].between(7, 19).astype(int)

    # 3. CYCLICAL LAGS (The 'Memory' - Critical for XGBoost)
    # We use 'shift' based on your 15-min intervals (4 points per hour)
    points_per_hour = 4 
    points_per_day = points_per_hour * 24
    points_per_week = points_per_day * 7
    
    # Lag 1: Exactly one week ago (Same day, same time)
    feat_df['lag_7d'] = feat_df['consumo_kwh'].shift(points_per_week)
    
    # Lag 2: Rolling average of the last 24 hours (Detects recent trends)
    feat_df['rolling_24h_mean'] = feat_df['consumo_kwh'].rolling(window=points_per_day).mean()

    # 4. CLIMATE INTEGRATION
    # If you have a separate temperature file, we merge it here
    if temp_df is not None:
        feat_df = pd.merge_asof(feat_df, temp_df, on='fecha')
        # The 'V-Shape' math:
        if 'temp' in feat_df.columns:
            feat_df['temp_sq'] = feat_df['temp'] ** 2
            # Interaction: How temperature affects the building during business hours vs night
            feat_df['temp_occupancy_interact'] = feat_df['temp'] * feat_df['is_business_hours']

    # Remove the 'NaN' rows created by the shifts (the first week of data)
    return feat_df.dropna().reset_index(drop=True)
