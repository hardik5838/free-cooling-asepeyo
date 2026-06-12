import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. CORE THERMODYNAMIC LOGIC ENGINE
# ==========================================

def calculate_target_launch_temp(t_comfort_max, delta_t_p1, r_drift, floor_temp=19.0):
    """
    Calculates the exact target launch temperature (X) required at 08:59 
    to coast through the P1 price spike without mechanical cooling.
    Formula: X = T_comfort_max - (Delta_t_P1 * R_drift)
    """
    calculated_x = t_comfort_max - (delta_t_p1 * r_drift)
    # Apply safety floor boundary constraint to prevent localized over-cooling
    return max(calculated_x, floor_temp)

def run_precool_simulation(hours, temps, config):
    """
    Simulates the 24-hour baseline vs optimized thermal trajectory of the building.
    """
    df = pd.DataFrame({'hora': hours, 'temp_ambient': temps})
    
    # Define Spanish PVPC Tariff Prices (P1: Peak, P2: Flat, P3: Valley)
    def get_tariff_price(h):
        if 10 <= h < 14 or 18 <= h < 22: 
            return 0.30  # P1 Peak (€/kWh)
        if 8 <= h < 10 or 14 <= h < 18 or 22 <= h < 24: 
            return 0.18  # P2 Mid (€/kWh)
        return 0.10      # P3 Valley (€/kWh)
        
    df['precio_kwh'] = df['hora'].apply(get_tariff_price)
    
    # Initialize Tracking Arrays
    temp_in_baseline = []
    temp_in_optimized = []
    hvac_kw_baseline = []
    hvac_kw_optimized = []
    
    # Initial state conditions at midnight
    t_in_base = config['t_setpoint']
    t_in_opt = config['t_setpoint']
    
    # Step through every hour of the day
    for _, row in df.iterrows():
        h = row['hora']
        t_out = row['temp_ambient']
        
        # --- 1. BASELINE SIMULATION (Standard Fixed Schedule) ---
        # Traditional HVAC runs strictly inside office hours (e.g., 08:00 - 20:00)
        if 8 <= h < 20:
            hvac_base = np.clip((config['ua'] / 1000.0) * (t_out - config['t_setpoint']) + config['q_int'], 0, config['max_cap'])
            t_in_base = config['t_setpoint']
        else:
            hvac_base = 0.0
            # Natural thermal drift when HVAC is OFF overnight
            q_trans = (config['ua'] / 1000.0) * (t_out - t_in_base)
            t_in_base += (q_trans + config['q_int']) / config['capacitance']
            
        # --- 2. OPTIMIZED SIMULATION (Glider Theory Pre-cooling) ---
        # P3 Pre-cool execution window (04:00 - 09:00)
        if 4 <= h < 9:
            hvac_opt = config['max_cap']  # Max out cooling using cheap valley power
            q_trans = (config['ua'] / 1000.0) * (t_out - t_in_opt)
            t_in_opt += (q_trans + config['q_int'] - hvac_opt * config['cop']) / config['capacitance']
            t_in_opt = max(t_in_opt, config['t_launch_target']) # Bound by target
            
        # P1 Shutdown Window (10:00 - 14:00) -> Mechanical Cooldown completely frozen
        elif 10 <= h < 14 and t_in_opt < config['t_comfort_max']:
            hvac_opt = 0.0
            q_trans = (config['ua'] / 1000.0) * (t_out - t_in_opt)
            t_in_opt += (q_trans + config['q_int']) / config['capacitance']
            
        # Standard Operating / Maintenance Window
        elif 8 <= h < 20:
            hvac_opt = np.clip((config['ua'] / 1000.0) * (t_out - config['t_setpoint']) + config['q_int'], 0, config['max_cap'])
            t_in_opt = config['t_setpoint']
        else:
            hvac_opt = 0.0
            q_trans = (config['ua'] / 1000.0) * (t_out - t_in_opt)
            t_in_opt += (q_trans + config['q_int']) / config['capacitance']
            
        # Save steps
        temp_in_baseline.append(t_in_base)
        temp_in_optimized.append(t_in_opt)
        hvac_kw_baseline.append(hvac_base)
        hvac_kw_optimized.append(hvac_opt)
        
    df['t_in_baseline'] = temp_in_baseline
    df['t_in_optimized'] = temp_in_optimized
    df['hvac_kw_baseline'] = hvac_kw_baseline
    df['hvac_kw_optimized'] = hvac_kw_optimized
    
    # Cost Calculations
    df['cost_baseline'] = df['hvac_kw_baseline'] * df['precio_kwh']
    df['cost_optimized'] = df['hvac_kw_optimized'] * df['precio_kwh']
    
    return df

# ==========================================
# 2. STREAMLIT INTERFACE LAYER
# ==========================================

def show_precool_optimizer_page():
    st.title("🧊 Oasis Pre-cooling Optimizer")
    st.markdown("Transform your building profile from a passive consumer into an active thermal asset using **Glider Theory** structural charging.")
    
    # Create Layout Columns
    col_inputs, col_results = st.columns([1, 2])
    
    with col_inputs:
        st.header("⚙️ Building & Target Parameters")
        
        # Thermal Watchdog Inputs
        t_comfort_max = st.slider("Max Legal Comfort Limit (T_comfort_max) [°C]", 24.0, 28.0, 27.0, 
                                  help="Maximum legally mandated thermal comfort limit in Spain.")
        delta_t_p1 = st.number_input("P1 Peak Duration (Δt_P1) [Hours]", value=4.0, step=0.5,
                                    help="Temporal duration of the maximum high-cost P1 tariff window (10:00 - 14:00).")
        r_drift = st.slider("Characteristic Building Rise Rate (R_drift) [°C/hr]", 0.5, 3.0, 1.6, step=0.1,
                            help="How fast your building spaces warm up naturally when mechanical systems are frozen.")
        floor_temp = st.number_input("Pre-cool Safety Floor Threshold [°C]", value=19.0, min_value=16.0,
                                     help="Prevents over-cooling to secure occupant wellness before arrival.")
        
        # Calculate dynamic target point using Thesis Formula (1)
        t_launch_target = calculate_target_launch_temp(t_comfort_max, delta_t_p1, r_drift, floor_temp)
        
        st.metric(label="🎯 Calculated Target Launch Temperature (X)", value=f"{t_launch_target:.1f} °C")
        
        st.divider()
        st.header("🏢 Physical Structural Mass Attributes")
        ua = st.number_input("Building Heat Loss Envelope Coefficient (U × A) [W/K]", value=1200.0, step=100.0)
        capacitance = st.number_input("Building Thermal Capacitance (C) [kJ/K]", value=45000.0, step=1000.0)
        max_cap = st.number_input("Max Electrical HVAC Capacity [kW]", value=35.0, step=5.0)
        cop = st.number_input("Coefficient of Performance (COP)", value=3.5, step=0.1)
        q_int = st.slider("Internal Load Gains (Occupancy/Appliances) [kW]", 0.0, 20.0, 8.0)
        
    # Generate Synthetic/Simulated 24-Hour Hot Summer Air Profile
    hours = np.arange(0, 24, 1)
    temps_ambient = 22.0 + 12.0 * np.sin(np.pi * (hours - 6) / 12) # Peak heat at 15:00h
    
    config = {
        't_comfort_max': t_comfort_max, 't_setpoint': 22.0, 't_launch_target': t_launch_target,
        'ua': ua, 'capacitance': capacitance, 'max_cap': max_cap, 'cop': cop, 'q_int': q_int
    }
    
    # Process Metrics Engine
    df_sim = run_precool_simulation(hours, temps_ambient, config)
    
    with col_results:
        # Calculate Financial KPI Deliverables
        total_cost_base = df_sim['cost_baseline'].sum()
        total_cost_opt = df_sim['cost_optimized'].sum()
        savings_daily = total_cost_base - total_cost_opt
        savings_pct = (savings_daily / total_cost_base) * 100 if total_cost_base > 0 else 0
        
        st.header("📊 Optimization Financial KPIs")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Baseline Daily Cost", f"{total_cost_base:.2f} €")
        kpi2.metric("Optimized Daily Cost", f"{total_cost_opt:.2f} €", delta=f"-{savings_daily:.2f} €")
        kpi3.metric("HVAC Efficiency Shift", f"{savings_pct:.1f} %", delta_color="normal")
        
        st.divider()
        
        # Chart 1: Electric Profile Performance Comparison
        st.subheader("⚡ Load Shifting Performance Strategy")
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['hvac_kw_baseline'], name="Baseline Schedule Load", line=dict(color='rgb(231, 76, 60)', dash='dash', width=2)))
        fig_load.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['hvac_kw_optimized'], name="Oasis Pre-cooled (Shifted) Load", fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)', line=dict(color='rgb(46, 204, 113)', width=3)))
        fig_load.update_layout(xaxis=dict(title="Hour of the Day", dtick=2), yaxis=dict(title="HVAC Power Consumption (kW)"), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_load, use_container_width=True)
        
        # Chart 2: Internal Space Ambient Curve Flyby
        st.subheader("🌡️ Internal Building Temperature Glider Space Trajectory")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['temp_ambient'], name="Outdoor Temperature", line=dict(color='orange', width=2, dash='dot')))
        fig_temp.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['t_in_baseline'], name="Baseline Space Room Temp", line=dict(color='grey', width=2)))
        fig_temp.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['t_in_optimized'], name="Optimized Thermal Trajectory", line=dict(color='blue', width=3)))
        # Draw explicit lines indicating lower and upper comfort limit fields
        fig_temp.add_hline(y=t_comfort_max, line_dash="dash", line_color="red", annotation_text="Comfort Ceiling Cap")
        fig_temp.add_hline(y=t_launch_target, line_dash="dash", line_color="green", annotation_text="Pre-cool Glide Target Launch")
        fig_temp.update_layout(xaxis=dict(title="Hour of the Day", dtick=2), yaxis=dict(title="Temperature (°C)"), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_temp, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="Oasis Pre-cooling Framework", layout="wide")
    show_precool_optimizer_page()
