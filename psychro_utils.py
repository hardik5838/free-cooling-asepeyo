## psychro_utils.py
import streamlit as st
import psychrolib as psy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Configurar la biblioteca psicrométrica para usar unidades del Sistema Internacional (SI)
psy.SetUnitSystem(psy.SI)

@st.cache_data
def calculate_psychrometrics(tdb, rel_hum, pressure=101325):
    """Calcula las propiedades psicrométricas a partir de Tdb y RH para un solo punto."""
    if rel_hum is None or tdb is None or pd.isna(tdb) or pd.isna(rel_hum):
        return None
    
    # Asegurarse que RH está en fracción (0-1)
    if rel_hum > 1.0:
        rel_hum = rel_hum / 100.0

    try:
        # Tdb (Dry-bulb temp), HumRatio (Humidity ratio), TDewPoint (Dew-point temp), 
        # RelHum (Relative humidity), Enthalpy
        results = psy.CalcPsychrometricsFromRelHum(tdb, rel_hum, pressure)
        hum_ratio = results[0]
        t_dew_point = results[1]
        enthalpy = results[3] / 1000 # Convertir de J/kg a kJ/kg
        
        return {
            "Tdb": tdb, 
            "RH": rel_hum * 100, 
            "HumRatio_g_kg": hum_ratio * 1000, # Convertir de kg/kg a g/kg
            "TDewPoint": t_dew_point, 
            "Enthalpy_kJ_kg": enthalpy
        }
    except Exception:
        return None

def get_base_psychro_fig():
    """Dibuja el fondo del gráfico psicrométrico (líneas de RH)."""
    fig = go.Figure()
    temp_range = np.linspace(-10, 50, 61)
    pressure = 101325 # Presión estándar al nivel del mar

    # 1. Línea de Saturación (100% RH)
    hum_ratio_100 = [psy.CalcPsychrometricsFromRelHum(t, 1.0, pressure)[0] * 1000 for t in temp_range]
    fig.add_trace(go.Scatter(
        x=temp_range, y=hum_ratio_100, 
        mode='lines', name='100% RH', 
        line=dict(color='blue', width=3)
    ))

    # 2. Líneas de RH constantes
    for rh in [80, 60, 40, 20]:
        hum_ratio_rh = [psy.CalcPsychrometricsFromRelHum(t, rh / 100.0, pressure)[0] * 1000 for t in temp_range]
        fig.add_trace(go.Scatter(
            x=temp_range, y=hum_ratio_rh, 
            mode='lines', name=f'{rh}% RH', 
            line=dict(color='rgba(100, 100, 100, 0.5)', width=1, dash='dot')
        ))

    fig.update_layout(
        title="Gráfico Psicrométrico Interactivo",
        xaxis_title="Temperatura de Bulbo Seco (Tdb) - °C",
        yaxis_title="Relación de Humedad (g vapor / kg aire seco)",
        xaxis=dict(range=[-10, 40]), 
        yaxis=dict(range=[0, 30]), 
        height=600,
        legend_title="Leyenda"
    )
    return fig

@st.cache_data
def get_realtime_weather(latitude, longitude):
    """Obtiene el clima actual de la API de Open-Meteo."""
    try:
        # Configurar el cliente de Open-Meteo
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["temperature_2m", "relative_humidity_2m"]
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        current = response.Current()
        
        return {
            "Tdb": current.Variables(0).Value(),
            "RH": current.Variables(1).Value()
        }
    except Exception as e:
        st.error(f"Error al contactar la API del clima: {e}")
        return None

@st.cache_data
def load_and_process_csv(uploaded_file, col_tdb_name, col_rh_name):
    """Carga un CSV y calcula las propiedades psicrométricas para cada fila."""
    df = pd.read_csv(uploaded_file)
    
    # Asegurar que los datos son numéricos
    df[col_tdb_name] = pd.to_numeric(df[col_tdb_name], errors='coerce')
    df[col_rh_name] = pd.to_numeric(df[col_rh_name], errors='coerce')
    df = df.dropna(subset=[col_tdb_name, col_rh_name])

    # Aplicar la función de psicrometría a cada fila
    # Esto crea una nueva columna 'PsyData' que contiene diccionarios
    df['PsyData'] = df.apply(
        lambda row: calculate_psychrometrics(row[col_tdb_name], row[col_rh_name]), 
        axis=1
    )
    
    # Descartar filas donde el cálculo falló
    df = df.dropna(subset=['PsyData'])
    
    # "Explotar" el diccionario en columnas separadas
    psy_df = pd.json_normalize(df['PsyData'])
    
    # Unir de nuevo con los datos originales (si se quiere)
    df_final = pd.concat([df.reset_index(drop=True), psy_df.reset_index(drop=True)], axis=1)
    return df_final

def check_free_cooling_potential(props_ext, props_int, rules):
    """Verifica si un punto exterior cumple con las reglas de Free Cooling."""
    if not props_ext or not props_int:
        return "N/A"

    # 1. Reglas directas (caja)
    rule_t_min = props_ext['Tdb'] > rules['t_min']
    rule_t_max = props_ext['Tdb'] < rules['t_max']
    rule_dp_max = props_ext['TDewPoint'] < rules['dp_max']
    
    # 2. Regla de Entalpía (Economizador)
    rule_enthalpy = props_ext['Enthalpy_kJ_kg'] < props_int['Enthalpy_kJ_kg']

    if rule_t_min and rule_t_max and rule_dp_max and rule_enthalpy:
        return "✅ Potencial de Free Cooling"
    else:
        return "❌ Sin Potencial"
