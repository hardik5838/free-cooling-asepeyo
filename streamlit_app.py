## app.py

import streamlit as st

import plotly.graph_objects as go

import psychrolib as psy

import numpy as np

import time



# Importamos nuestras funciones de API personalizadas

import aranet_utils



# Configurar la biblioteca psicrométrica para usar unidades del Sistema Internacional (SI)

psy.SetUnitSystem(psy.SI)



# --- Configuración de la Página ---

st.set_page_config(

    page_title="Análisis de Free Cooling (con API)",

    page_icon="❄️",

    layout="wide"

)



st.title("❄️ Aplicación de Análisis de Free Cooling (Conectada a Aranet)")

st.write("""

Esta aplicación carga datos en vivo desde tu API de Aranet. 

Selecciona tus sensores de aire interior y exterior en la barra lateral 

para analizar el potencial de 'free cooling'.

""")



# --- Funciones de Cálculo y Gráfico (Las mismas de la versión manual) ---



@st.cache_data

def calculate_psychrometrics(tdb, rel_hum, pressure=101325):

    """Calcula las propiedades psicrométricas a partir de Tdb y RH."""

    if rel_hum is None or tdb is None:

        return None

    if rel_hum > 1.0:

        rel_hum = rel_hum / 100.0



    try:

        results = psy.CalcPsychrometricsFromRelHum(tdb, rel_hum, pressure)

        hum_ratio = results[0]

        t_dew_point = results[1]

        enthalpy = results[3] / 1000

        return {

            "Tdb": tdb, "RH": rel_hum * 100, "HumRatio_g_kg": hum_ratio * 1000,

            "TDewPoint": t_dew_point, "Enthalpy_kJ_kg": enthalpy

        }

    except Exception:

        return None



@st.cache_data

def plot_psychrometric_chart(internal_props, external_props):

    """Dibuja el gráfico psicrométrico."""

    fig = go.Figure()

    temp_range = np.linspace(-10, 50, 61)

    

    # 1. Línea de Saturación (100% RH)

    hum_ratio_100 = [psy.CalcPsychrometricsFromRelHum(t, 1.0, 101325)[0] * 1000 for t in temp_range]

    fig.add_trace(go.Scatter(x=temp_range, y=hum_ratio_100, mode='lines', name='100% RH', line=dict(color='blue', width=3)))



    # 2. Líneas de RH constantes

    for rh in [80, 60, 40, 20]:

        hum_ratio_rh = [psy.CalcPsychrometricsFromRelHum(t, rh / 100.0, 101325)[0] * 1000 for t in temp_range]

        fig.add_trace(go.Scatter(x=temp_range, y=hum_ratio_rh, mode='lines', name=f'{rh}% RH', line=dict(color='rgba(100, 100, 100, 0.5)', width=1, dash='dot')))

    

    # 3. Punto de Aire Interior

    if internal_props:

        fig.add_trace(go.Scatter(

            x=[internal_props['Tdb']], y=[internal_props['HumRatio_g_kg']],

            mode='markers+text', name='Aire Interior', text=["<b>Interior</b>"],

            textposition="top right", marker=dict(color='red', size=15, symbol='x')

        ))



    # 4. Punto de Aire Exterior

    if external_props:

        fig.add_trace(go.Scatter(

            x=[external_props['Tdb']], y=[external_props['HumRatio_g_kg']],

            mode='markers+text', name='Aire Exterior', text=["<b>Exterior</b>"],

            textposition="bottom right", marker=dict(color='green', size=15, symbol='circle')

        ))



    fig.update_layout(

        title="Gráfico Psicrométrico Interactivo",

        xaxis_title="Temperatura de Bulbo Seco (Tdb) - °C",

        yaxis_title="Relación de Humedad (g vapor / kg aire seco)",

        xaxis=dict(range=[-10, 40]), yaxis=dict(range=[0, 30]), height=600

    )

    return fig



# --- Carga de Datos de API ---

with st.spinner('Cargando datos de la API de Aranet...'):

    sensor_name_map = aranet_utils.load_sensors()

    measurements = aranet_utils.get_processed_measurements()



if not sensor_name_map or not measurements:

    st.error("No se pudieron cargar los datos de la API. Verifica tu clave en 'secrets.toml' y la conexión.")

    st.stop()



# --- Barra Lateral de Entradas (Inputs) ---

st.sidebar.header("Selección de Sensores 📡")



# Creamos la lista de nombres de sensores para los menús

sensor_names = list(sensor_name_map.keys())



# Menú para el sensor INTERIOR

selected_int_name = st.sidebar.selectbox(

    "1. Selecciona el Sensor INTERIOR",

    options=sensor_names,

    index=0 # Por defecto selecciona el primero de la lista

)

# Obtenemos el ID del sensor interior seleccionado

sensor_id_int = sensor_name_map[selected_int_name]



# Menú para el sensor EXTERIOR

selected_ext_name = st.sidebar.selectbox(

    "2. Selecciona el Sensor EXTERIOR",

    options=sensor_names,

    index=1 # Por defecto selecciona el segundo de la lista

)

# Obtenemos el ID del sensor exterior seleccionado

sensor_id_ext = sensor_name_map[selected_ext_name]



st.sidebar.info(f"Datos cargados para {len(measurements)} sensores.")

if st.sidebar.button("Recargar Datos"):

    st.cache_data.clear() # Limpia la caché

    st.rerun() # Vuelve a ejecutar la app



# --- Lógica Principal y Visualización ---



# Buscar los datos de los sensores seleccionados

data_int = measurements.get(sensor_id_int, {})

data_ext = measurements.get(sensor_id_ext, {})



# Obtener T y RH (con .get() para evitar errores si falta la métrica)

t_int_db = data_int.get("temperature")

rh_int = data_int.get("humidity")



t_ext_db = data_ext.get("temperature")

rh_ext = data_ext.get("humidity")



# Calcular propiedades

props_int = calculate_psychrometrics(t_int_db, rh_int)

props_ext = calculate_psychrometrics(t_ext_db, rh_ext)



# Dividir la página en columnas para métricas y recomendaciones

col_metrics, col_recommendation = st.columns([1, 1])



with col_metrics:

    st.subheader("Métricas Calculadas")

    if props_int and props_ext:

        col_int, col_ext = st.columns(2)

        with col_int:

            st.markdown(f"##### 🏠 INTERIOR ({selected_int_name})")

            st.metric("Temperatura", f"{props_int['Tdb']:.1f} °C")

            st.metric("Humedad", f"{props_int['RH']:.0f} %")

            st.metric("Entalpía (Energía)", f"{props_int['Enthalpy_kJ_kg']:.1f} kJ/kg")



        with col_ext:

            st.markdown(f"##### 🌳 EXTERIOR ({selected_ext_name})")

            st.metric("Temperatura", f"{props_ext['Tdb']:.1f} °C")

            st.metric("Humedad", f"{props_ext['RH']:.0f} %")

            st.metric("Entalpía (Energía)", f"{props_ext['Enthalpy_kJ_kg']:.1f} kJ/kg")

    else:

        st.warning("Uno o ambos sensores seleccionados no reportan Temperatura y Humedad.")



with col_recommendation:

    st.subheader("Recomendación de Free Cooling")

    if props_int and props_ext:

        temp_diff = props_int['Tdb'] - props_ext['Tdb']

        enthalpy_diff = props_int['Enthalpy_kJ_kg'] - props_ext['Enthalpy_kJ_kg']



        # Reglas para Free Cooling

        if (temp_diff > 2) and (enthalpy_diff > 4): 

            st.success("✅ POTENCIAL DE FREE COOLING DETECTADO")

            st.write(f"El aire exterior está **{temp_diff:.1f} °C más frío** y tiene **{enthalpy_diff:.1f} kJ/kg menos energía**.")

            st.write("**Acción:** Se puede utilizar el aire exterior para enfriar.")

        else:

            st.error("❌ FREE COOLING NO RECOMENDADO")

            if temp_diff <= 2:

                st.write("Razón: El aire exterior no está lo suficientemente frío.")

            elif enthalpy_diff <= 4:

                st.write("Razón: El aire exterior es demasiado húmedo (alta energía).")

    else:

        st.info("Esperando datos válidos de T/RH para ambos sensores...")



# Mostrar el gráfico psicrométrico

if props_int and props_ext:

    st.plotly_chart(plot_psychrometric_chart(props_int, props_ext), use_container_width=True)

else:

    st.error("No se puede dibujar el gráfico. Asegúrate de que los sensores seleccionados tengan datos de T y RH.")


