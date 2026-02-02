import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import io
import os
from urllib.parse import quote
from scipy.stats import weibull_min
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. CORE PHYSICS MODELS (Oasis White)
# ==========================================

def oasis_white_model(total_daily_kwh, u_value=0.5, temp_out=30, temp_set=22, 
                      w_shape=5.0, w_scale=14.0, ai_dampening=1.0):
    t = np.linspace(0, 24, 100)
    
    # 1. Weibull Occupancy
    occupancy_raw = weibull_min.pdf(t, w_shape, loc=0, scale=w_scale)
    occupancy = (occupancy_raw / np.max(occupancy_raw)) * ai_dampening
    
    # 2. Floor Area Derivation
    avg_intensity = 0.05 
    floor_area = total_daily_kwh / (avg_intensity * 24)
    
    # 3. Lighting
    avg_illumination_per_m2 = 0.010 
    control_factor = 0.8
    lighting_active = floor_area * avg_illumination_per_m2 * control_factor * occupancy
    lighting = (lighting_active * 0.85) + (lighting_active.max() * 0.15)

    # 4. Ventilation
    del_t = abs(temp_out - temp_set)
    cop_vent = 3.5 
    vent_per_person = (occupancy * 9 + 1)
    vent_base = vent_per_person * (floor_area / 15)
    ventilation = (vent_base * 0.95) + (vent_base.max() * 0.05) + (del_t / cop_vent)

    # 5. HVAC & Others
    hvac_load_max = (u_value * floor_area * del_t) / 1000 
    hvac = (occupancy * 0.95 * hvac_load_max) + (0.05 * hvac_load_max)
    others = (occupancy * 0.95 * (total_daily_kwh/24 * 0.2)) + (0.05 * (total_daily_kwh/24 * 0.2))

    return t, lighting, ventilation, hvac, others, floor_area

# ==========================================
# 2. OPTIMIZATION & CALIBRATION ENGINE
# ==========================================

def generate_load_curve(hours, start, end, max_kw, nominal_pct=1.0, residual_pct=0.0):
    curve = np.zeros(len(hours))
    for i, h in enumerate(hours):
        activity_val = 0.0
        if start <= end:
            if start <= h < end: activity_val = 1.0
        else:
            if h >= start or h < end: activity_val = 1.0
        
        val = residual_pct + activity_val * (nominal_pct - residual_pct)
        curve[i] = val * max_kw
    return curve

def run_simulation(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    df['sim_base'] = generate_load_curve(hours, 0, 24, config['base_kw'], 1.0, 1.0)
    
    delta_T = np.maximum(0, df['temperatura_c'] - config['hvac_setpoint']) # Cooling mode
    thermal_load_raw = (config['hvac_ua'] * delta_T) 
    hvac_avail = generate_load_curve(hours, config['hvac_s'], config['hvac_e'], 1.0, 1.0, config['hvac_res'])
    df['sim_therm'] = np.clip(thermal_load_raw, 0, config['hvac_kw']) * hvac_avail

    df['sim_ops'] = generate_load_curve(hours, config['ops_s'], config['ops_e'], config['ops_kw'], 1.0, 0.0)
    df['sim_total'] = df['sim_base'] + df['sim_therm'] + df['sim_ops']
    return df

def objective_function(params, df_real):
    config = {
        'base_kw': params[0], 'hvac_kw': params[1], 'hvac_s': int(params[2]),
        'hvac_e': int(params[3]), 'hvac_ua': params[4], 'hvac_setpoint': 22.0,
        'hvac_res': 0.1, 'ops_kw': params[5], 'ops_s': int(params[6]), 'ops_e': int(params[7]),
    }
    df_sim = run_simulation(df_real, config)
    return np.sqrt(mean_squared_error(df_real['consumo_kwh'], df_sim['sim_total']))

def run_auto_calibration(df_avg):
    max_kwh = df_avg['consumo_kwh'].max()
    bounds = [(0, max_kwh*0.5), (0, max_kwh), (4, 10), (16, 22), (0.1, 5.0), (0, max_kwh), (6, 10), (17, 21)]
    result = differential_evolution(objective_function, bounds, args=(df_avg,), maxiter=15, popsize=10, seed=42)
    return result.x

# ==========================================
# 3. DATA LOADING & UI
# ==========================================

@st.cache_data
def load_energy_data(file_input):
    try:
        df = pd.read_csv(file_input, sep=',', skipinitialspace=True)
        df.columns = df.columns.str.strip()
        col_Fecha, col_energia = 'Fecha', 'Energía activa (kWh)'
        
        if col_Fecha not in df.columns or col_energia not in df.columns:
            return pd.DataFrame()
        
        df = df.rename(columns={col_Fecha: 'Fecha', col_energia: 'consumo_kwh'})
        if df['consumo_kwh'].dtype == 'object':
            df['consumo_kwh'] = df['consumo_kwh'].astype(str).str.replace(',', '.')
        
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['Fecha', 'consumo_kwh']).sort_values('Fecha')
    except:
        return pd.DataFrame()

@st.cache_data
def load_weather_data(file_path):
    try:
        if isinstance(file_path, str) and file_path.startswith('http'):
            content = requests.get(file_path).text
        else:
            content = file_path.getvalue().decode("utf-8")
        
        lines = content.splitlines()
        start_row = next(i for i, line in enumerate(lines) if "YEAR,MO,DY,HR" in line)
        df = pd.read_csv(io.StringIO('\n'.join(lines[start_row:])))
        df['Fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
        df.rename(columns={'T2M': 'temperatura_c', 'RH2M': 'humedad_relativa'}, inplace=True)
        return df[['Fecha', 'temperatura_c', 'humedad_relativa']]
    except:
        return pd.DataFrame()

def show_nilm_page(df_consumo, df_clima):
    st.subheader("🤖 Digital Twin & Auto-Calibration")
    
    if df_consumo.empty:
        st.warning("Please upload energy data first.")
        return

    # Prep average day
    df_merged = pd.merge(df_consumo, df_clima, on='Fecha', how='inner') if not df_clima.empty else df_consumo.assign(temperatura_c=22.0)
    df_avg = df_merged.groupby(df_merged['Fecha'].dt.hour).agg({'consumo_kwh': 'mean', 'temperatura_c': 'mean'}).reset_index().rename(columns={'Fecha': 'hora'})

    if 'opt_params' not in st.session_state:
        st.session_state['opt_params'] = [10.0, 20.0, 8.0, 18.0, 1.0, 15.0, 8.0, 19.0]

    with st.sidebar:
        if st.button("⚡ Auto-Calibrate Model", type="primary"):
            with st.spinner("AI is solving building DNA..."):
                st.session_state['opt_params'] = run_auto_calibration(df_avg)

        p = st.session_state['opt_params']
        base_kw = st.slider("Base Load (kW)", 0.0, 200.0, float(p[0]))
        h_kw = st.slider("HVAC Max (kW)", 0.0, 500.0, float(p[1]))
        h_s = st.slider("HVAC Start", 0, 24, int(p[2]))
        h_e = st.slider("HVAC End", 0, 24, int(p[3]))
        h_ua = st.slider("Thermal Sensitivity (UA)", 0.1, 10.0, float(p[4]))
        o_kw = st.slider("Ops Max (kW)", 0.0, 500.0, float(p[5]))
        o_s = st.slider("Ops Start", 0, 24, int(p[6]))
        o_e = st.slider("Ops End", 0, 24, int(p[7]))

    config = {'base_kw': base_kw, 'hvac_kw': h_kw, 'hvac_s': h_s, 'hvac_e': h_e, 'hvac_ua': h_ua, 'hvac_setpoint': 21.0, 'hvac_res': 0.1, 'ops_kw': o_kw, 'ops_s': o_s, 'ops_e': o_e}
    df_sim = run_simulation(df_avg, config)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], name='REAL Meter', line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'], name='Digital Twin', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_base'], name='Base', stackgroup='one', fillcolor='rgba(100,100,100,0.2)'))
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_therm'], name='HVAC', stackgroup='one', fillcolor='rgba(255,0,0,0.2)'))
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_ops'], name='Ops', stackgroup='one', fillcolor='rgba(255,200,0,0.2)'))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. MAIN APP ENTRY
# ==========================================

st.set_page_config(page_title="Asepeyo Energy Dashboard", layout="wide")

with st.sidebar:
    st.title("⚡ Control Panel")
    page = st.selectbox("Tool", ["Dashboard General", "Simulación NILM (Avanzado)", "Oasis Physics Model"])
    source = st.radio("Data Source", ["GitHub Demo", "Upload CSV"])

if source == "GitHub Demo":
    base_url = "https://raw.githubusercontent.com/hardik5838/EnergyPatternAnalysis/main/data/"
    df_consumo = load_energy_data(base_url + quote("251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv"))
    df_clima = load_weather_data(base_url + "weather.csv")
else:
    up_e = st.sidebar.file_uploader("Energy CSV")
    up_w = st.sidebar.file_uploader("Weather CSV")
    df_consumo = load_energy_data(up_e) if up_e else pd.DataFrame()
    df_clima = load_weather_data(up_w) if up_w else pd.DataFrame()

if page == "Dashboard General":
    st.title("Energy Consumption Patterns")
    if not df_consumo.empty:
        st.plotly_chart(px.line(df_consumo, x='Fecha', y='consumo_kwh', title="Raw Load Curve"), use_container_width=True)
    else:
        st.info("Upload data to view dashboard.")

elif page == "Simulación NILM (Avanzado)":
    show_nilm_page(df_consumo, df_clima)

elif page == "Oasis Physics Model":
    st.title("Oasis White Physics-Based Model")
    kwh_in = st.number_input("Total Daily kWh", value=5000)
    t, light, vent, hvac, misc, area = oasis_white_model(kwh_in)
    st.metric("Estimated Floor Area", f"{area:.2f} m²")
    
    fig_phys = go.Figure()
    fig_phys.add_trace(go.Scatter(x=t, y=light, name="Lighting"))
    fig_phys.add_trace(go.Scatter(x=t, y=vent, name="Ventilation"))
    fig_phys.add_trace(go.Scatter(x=t, y=hvac, name="HVAC"))
    st.plotly_chart(fig_phys, use_container_width=True)
