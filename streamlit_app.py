## app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import psychrolib as psy

# Importar nuestras nuevas funciones de utilidades
import psychro_utils as psy_utils

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Free Cooling",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ Aplicación de Análisis de Free Cooling")

# --- SELECCIÓN DE MODO ---
mode = st.radio(
    "Selecciona el modo de análisis:",
    ["Real-Time (API del Clima)", "Datos Históricos (Subir CSV)"],
    horizontal=True,
    label_visibility="collapsed"
)

# --- BARRA LATERAL (INPUTS COMUNES) ---
st.sidebar.header("1. Define tus Condiciones INTERNAS")
st.sidebar.write("Establece el punto de consigna deseado para tu edificio.")

# Usamos los valores que discutimos (19-25C, 40-60% RH)
t_int_db_slider = st.sidebar.slider(
    "Temperatura Interior (Tdb)", 
    min_value=18.0, max_value=26.0, value=22.0, step=0.5, format="%.1f °C"
)
rh_int_slider = st.sidebar.slider(
    "Humedad Interior (RH)", 
    min_value=30.0, max_value=70.0, value=50.0, step=1.0, format="%.0f %%"
)

st.sidebar.header("2. Define tus Reglas de Free Cooling")
st.sidebar.write("Establece los límites del AIRE EXTERIOR para activar el free cooling.")

# Usamos los valores por defecto que discutimos
fc_rule_t_min = st.sidebar.slider(
    "Temp. Exterior MÍNIMA (Tdb)",
    min_value=5.0, max_value=15.0, value=12.0, step=0.5, format="%.1f °C"
)
fc_rule_t_max = st.sidebar.slider(
    "Temp. Exterior MÁXIMA (Tdb)",
    min_value=15.0, max_value=25.0, value=18.0, step=0.5, format="%.1f °C"
)
fc_rule_dp_max = st.sidebar.slider(
    "Punto de Rocío Exterior MÁXIMO (TDewPoint)",
    min_value=10.0, max_value=16.0, value=13.0, step=0.5, format="%.1f °C"
)

# Empaquetar las reglas
fc_rules = {
    "t_min": fc_rule_t_min,
    "t_max": fc_rule_t_max,
    "dp_max": fc_rule_dp_max
}

# --- LÓGICA PRINCIPAL ---

# Calcular propiedades internas UNA VEZ
props_int = psy_utils.calculate_psychrometrics(t_int_db_slider, rh_int_slider / 100.0)

# -----------------------------------------------------------------
# FUNCIÓN PARA EL MODO REAL-TIME
# -----------------------------------------------------------------
def run_realtime_app(props_int, fc_rules):
    st.header("Análisis en Tiempo Real (API del Clima)")
    
    # Coordenadas de Barcelona por defecto
    col_lat, col_lon = st.columns(2)
    lat = col_lat.number_input("Latitud", value=41.38, format="%.4f")
    lon = col_lon.number_input("Longitud", value=2.17, format="%.4f")

    with st.spinner("Cargando datos del clima en tiempo real..."):
        weather_data = psy_utils.get_realtime_weather(lat, lon)
    
    if not weather_data:
        st.error("No se pudieron cargar los datos del clima.")
        st.stop()

    props_ext = psy_utils.calculate_psychrometrics(weather_data['Tdb'], weather_data['RH'])

    if not props_ext:
        st.error("Datos del clima inválidos recibidos de la API.")
        st.stop()

    # Dividir la página para métricas y recomendaciones
    col_metrics, col_recommendation = st.columns([1, 1])

    with col_metrics:
        st.subheader("Métricas Calculadas")
        col_int, col_ext = st.columns(2)
        with col_int:
            st.markdown("##### 🏠 INTERIOR (Objetivo)")
            st.metric("Temperatura", f"{props_int['Tdb']:.1f} °C")
            st.metric("Humedad", f"{props_int['RH']:.0f} %")
            st.metric("Entalpía", f"{props_int['Enthalpy_kJ_kg']:.1f} kJ/kg")
        with col_ext:
            st.markdown("##### 🌳 EXTERIOR (Actual)")
            st.metric("Temperatura", f"{props_ext['Tdb']:.1f} °C")
            st.metric("Humedad", f"{props_ext['RH']:.0f} %")
            st.metric("Entalpía", f"{props_ext['Enthalpy_kJ_kg']:.1f} kJ/kg")

    with col_recommendation:
        st.subheader("Recomendación de Free Cooling")
        status = psy_utils.check_free_cooling_potential(props_ext, props_int, fc_rules)
        
        if status == "✅ Potencial de Free Cooling":
            st.success(status)
            st.write("El aire exterior es adecuado para enfriar tu edificio.")
        else:
            st.error(status)
            st.write("El aire exterior no es óptimo. Se requiere A/C mecánico.")

    # Dibujar el gráfico
    fig = psy_utils.get_base_psychro_fig()
    
    # Añadir punto Interior
    fig.add_trace(go.Scatter(
        x=[props_int['Tdb']], y=[props_int['HumRatio_g_kg']],
        mode='markers+text', name='Aire Interior (Objetivo)', text=["<b>Interior</b>"],
        textposition="top right", marker=dict(color='red', size=15, symbol='x')
    ))
    
    # Añadir punto Exterior
    fig.add_trace(go.Scatter(
        x=[props_ext['Tdb']], y=[props_ext['HumRatio_g_kg']],
        mode='markers+text', name='Aire Exterior (Actual)', text=["<b>Exterior</b>"],
        textposition="bottom right", marker=dict(color='green', size=15, symbol='circle')
    ))
    
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------
# FUNCIÓN PARA EL MODO DE DATOS HISTÓRICOS
# -----------------------------------------------------------------
def run_past_data_app(props_int, fc_rules):
    st.header("Análisis de Datos Históricos (CSV)")
    
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV de datos climáticos", 
        type=["csv"]
    )
    
    if uploaded_file is None:
        st.info("Por favor, sube un archivo CSV para comenzar el análisis.")
        st.warning("Tu CSV debe tener al menos una columna de Temperatura y una de Humedad Relativa.")
        return

    st.subheader("Mapeo de Columnas")
    st.write("Indica a la aplicación qué columnas usar.")
    
    # Cargar una vista previa para la selección de columnas
    try:
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        available_cols = df_preview.columns.tolist()
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        col_tdb_name = st.selectbox("Selecciona la columna de Temperatura (Tdb)", available_cols, index=0)
    with col2:
        col_rh_name = st.selectbox("Selecciona la columna de Humedad (RH)", available_cols, index=min(1, len(available_cols)-1))

    if st.button("Procesar y Analizar Datos", type="primary"):
        with st.spinner("Cargando y procesando todo el archivo CSV..."):
            df = psy_utils.load_and_process_csv(uploaded_file, col_tdb_name, col_rh_name)
        
        if df.empty:
            st.error("No se pudieron procesar datos. Verifica tu CSV y la selección de columnas.")
            st.stop()
            
        st.success(f"Se procesaron {len(df)} filas de datos.")

        # 2. Aplicar las reglas de Free Cooling
        with st.spinner("Analizando potencial de free cooling para cada hora..."):
            df['FreeCooling'] = df.apply(
                lambda row: psy_utils.check_free_cooling_potential(row, props_int, fc_rules), 
                axis=1
            )
        
        # 3. Mostrar Resumen
        fc_counts = df['FreeCooling'].value_counts(normalize=True) * 100
        potential_pct = fc_counts.get("✅ Potencial de Free Cooling", 0.0)
        
        st.subheader("Resultados del Análisis")
        st.metric("Potencial de Free Cooling Disponible", f"{potential_pct:.1f} % del tiempo")

        # 4. Dibujar el Gráfico
        with st.spinner("Generando gráfico..."):
            fig = psy_utils.get_base_psychro_fig()

            # Separar datos para el gráfico
            df_yes = df[df['FreeCooling'] == "✅ Potencial de Free Cooling"]
            df_no = df[df['FreeCooling'] == "❌ Sin Potencial"]

            # Añadir puntos SIN potencial
            fig.add_trace(go.Scatter(
                x=df_no['Tdb'], y=df_no['HumRatio_g_kg'], mode='markers',
                name='Sin Potencial',
                marker=dict(color='orange', size=3, opacity=0.4)
            ))
            
            # Añadir puntos CON potencial
            fig.add_trace(go.Scatter(
                x=df_yes['Tdb'], y=df_yes['HumRatio_g_kg'], mode='markers',
                name='Free Cooling Disponible',
                marker=dict(color='green', size=4, opacity=0.7)
            ))
            
            # Añadir punto Interior
            fig.add_trace(go.Scatter(
                x=[props_int['Tdb']], y=[props_int['HumRatio_g_kg']],
                mode='markers+text', name='Aire Interior (Objetivo)', text=["<b>Interior</b>"],
                textposition="top right", marker=dict(color='red', size=15, symbol='x')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.head())

# --- Ejecutar el modo seleccionado ---
if mode == "Real-Time (API del Clima)":
    run_realtime_app(props_int, fc_rules)
else:
    run_past_data_app(props_int, fc_rules)
