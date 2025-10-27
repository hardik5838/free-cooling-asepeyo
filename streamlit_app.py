## app.py
import streamlit as st
import plotly.graph_objects as go
import psychrolib as psy
import numpy as np

# Configurar la biblioteca psicrométrica para usar unidades del Sistema Internacional (SI)
psy.SetUnitSystem(psy.SI)

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Free Cooling",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ Aplicación de Análisis de Free Cooling y Psicrometría")
st.write("""
Esta aplicación te ayuda a determinar el potencial de 'free cooling' comparando
las condiciones del aire interior y exterior en un gráfico psicrométrico.
""")

# --- Funciones de Cálculo y Gráfico ---

def calculate_psychrometrics(tdb, rel_hum, pressure=101325):
    """
    Calcula las propiedades psicrométricas a partir de Tdb y RH.
    Devuelve un diccionario con las propiedades clave.
    """
    if rel_hum > 1.0:
        rel_hum = rel_hum / 100.0  # Asegurarse de que RH esté en fracción (ej: 0.5)

    try:
        # Calcular todas las propiedades desde Tdb y RH
        results = psy.CalcPsychrometricsFromRelHum(tdb, rel_hum, pressure)

        # Extraer los valores (HumRatio, TDewPoint, TWetBulb, Enthalpy, etc.)
        hum_ratio = results[0]      # kg_vapor / kg_aire_seco
        t_dew_point = results[1]    # °C
        t_wet_bulb = results[2]     # °C
        enthalpy = results[3] / 1000  # J/kg -> kJ/kg

        return {
            "Tdb": tdb,
            "RH": rel_hum * 100,
            "HumRatio_kg_kg": hum_ratio,
            "HumRatio_g_kg": hum_ratio * 1000, # Convertir a g/kg para el gráfico
            "TDewPoint": t_dew_point,
            "TWetBulb": t_wet_bulb,
            "Enthalpy_kJ_kg": enthalpy
        }
    except Exception as e:
        st.error(f"Error en el cálculo psicrométrico: {e}")
        return None

def plot_psychrometric_chart(internal_props, external_props):
    """
    Dibuja el gráfico psicrométrico con los puntos de aire interior y exterior.
    """
    fig = go.Figure()

    # Rango de temperaturas para dibujar las líneas
    temp_range = np.linspace(-10, 50, 61)

    # 1. Dibujar la Línea de Saturación (100% RH)
    hum_ratio_100 = [psy.CalcPsychrometricsFromRelHum(t, 1.0, 101325)[0] * 1000 for t in temp_range]
    fig.add_trace(go.Scatter(
        x=temp_range,
        y=hum_ratio_100,
        mode='lines',
        name='100% RH (Saturación)',
        line=dict(color='blue', width=3)
    ))

    # 2. Dibujar líneas de Humedad Relativa (RH) constantes
    for rh in [80, 60, 40, 20]:
        rh_fraction = rh / 100.0
        hum_ratio_rh = [psy.CalcPsychrometricsFromRelHum(t, rh_fraction, 101325)[0] * 1000 for t in temp_range]
        fig.add_trace(go.Scatter(
            x=temp_range,
            y=hum_ratio_rh,
            mode='lines',
            name=f'{rh}% RH',
            line=dict(color='rgba(100, 100, 100, 0.5)', width=1, dash='dot')
        ))
    
    # 3. Añadir punto de Aire Interior
    if internal_props:
        fig.add_trace(go.Scatter(
            x=[internal_props['Tdb']],
            y=[internal_props['HumRatio_g_kg']],
            mode='markers+text',
            name='Aire Interior',
            text=["<b>Interior</b>"],
            textposition="top right",
            marker=dict(color='red', size=15, symbol='x')
        ))

    # 4. Añadir punto de Aire Exterior
    if external_props:
        fig.add_trace(go.Scatter(
            x=[external_props['Tdb']],
            y=[external_props['HumRatio_g_kg']],
            mode='markers+text',
            name='Aire Exterior',
            text=["<b>Exterior</b>"],
            textposition="bottom right",
            marker=dict(color='green', size=15, symbol='circle')
        ))

    # --- Configuración del Layout del Gráfico ---
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

# --- Barra Lateral de Entradas (Inputs) ---
st.sidebar.header("Parámetros del Aire 💨")

st.sidebar.subheader("1. Aire Interior (Sensores)")
t_int_db = st.sidebar.slider("Temperatura Interior (Tdb)", min_value=15.0, max_value=35.0, value=24.0, step=0.5)
rh_int = st.sidebar.slider("Humedad Relativa Interior (RH)", min_value=0, max_value=100, value=50, step=1)

st.sidebar.subheader("2. Aire Exterior (Datos API)")
st.sidebar.info("Estos valores vendrán de tu API 'Hardik Freecooling' cuando tengamos la URL.")
t_ext_db = st.sidebar.slider("Temperatura Exterior (Tdb)", min_value=-10.0, max_value=50.0, value=15.0, step=0.5)
rh_ext = st.sidebar.slider("Humedad Relativa Exterior (RH)", min_value=0, max_value=100, value=60, step=1)

# --- Lógica Principal y Visualización ---

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
            st.markdown("##### 🏠 INTERIOR")
            st.metric("Entalpía (Energía)", f"{props_int['Enthalpy_kJ_kg']:.1f} kJ/kg")
            st.metric("Punto de Rocío (Tdp)", f"{props_int['TDewPoint']:.1f} °C")

        with col_ext:
            st.markdown("##### 🌳 EXTERIOR")
            st.metric("Entalpía (Energía)", f"{props_ext['Enthalpy_kJ_kg']:.1f} kJ/kg")
            st.metric("Punto de Rocío (Tdp)", f"{props_ext['TDewPoint']:.1f} °C")

with col_recommendation:
    st.subheader("Recomendación de Free Cooling")
    if props_int and props_ext:
        
        # Condiciones para Free Cooling:
        # 1. Aire exterior MÁS FRÍO que el interior.
        # 2. Aire exterior con MENOS ENERGÍA (Entalpía) que el interior.
        # 3. (Opcional) Aire exterior por encima de un mínimo (ej. 13°C) para no enfriar demasiado.
        
        temp_diff = props_int['Tdb'] - props_ext['Tdb']
        enthalpy_diff = props_int['Enthalpy_kJ_kg'] - props_ext['Enthalpy_kJ_kg']

        if temp_diff > 2 and enthalpy_diff > 4: # Umbrales (ej: 2°C y 4 kJ/kg de diferencia)
            st.success("✅ POTENCIAL DE FREE COOLING DETECTADO")
            st.write(f"""
            El aire exterior está **{temp_diff:.1f} °C más frío** y tiene **{enthalpy_diff:.1f} kJ/kg menos energía** que el aire interior.
            
            **Acción:** Se puede utilizar el aire exterior para enfriar el edificio y ahorrar energía.
            """)
        else:
            st.error("❌ FREE COOLING NO RECOMENDADO")
            st.write("""
            Las condiciones exteriores no son favorables (aire muy caliente o muy húmedo). 
            Se requiere enfriamiento mecánico (Compresores).
            """)

# Mostrar el gráfico psicrométrico
st.plotly_chart(plot_psychrometric_chart(props_int, props_ext), use_container_width=True)
