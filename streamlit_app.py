## app.py
import streamlit as st
import requests
import json
import pandas as pd

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Explorador de API Aranet",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Explorador de API de Aranet Cloud (V2)")
st.write("""
Presiona los botones para conectarte a tu API. Hemos actualizado el 
encabezado de autenticación a 'ApiKey' según la documentación.
""")

# --- Configuración de la API ---
# Carga la clave de API desde el archivo de secretos
try:
    API_KEY = "x5cuvkj5q342627bawvwkrjgd85z4fvd"

except KeyError:
    st.error("Error: No se encontró la 'ARANET_API_KEY' en el archivo .streamlit/secrets.toml")
    st.error("Asegúrate de que el archivo existe y la clave está guardada.")
    st.stop()

API_BASE_URL = "https://aranet.cloud/api/v1"

# --- LÍNEA CORREGIDA ---
# Basado en la documentación 'openapi' (Swagger), el encabezado
# que se espera es "ApiKey", no "Authorization".
API_HEADERS = {
    "ApiKey": API_KEY,  # <--- ¡Este es el cambio!
    "Accept": "application/json"
}
# -------------------------

# --- Funciones de Carga de Datos (sin cambios) ---

@st.cache_data(ttl=600) # Cachear los sensores por 10 minutos
def load_aranet_sensors(headers):
    """Carga la lista de todos los sensores."""
    try:
        url = f"{API_BASE_URL}/sensors"
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Lanza un error si la petición falla
        return response.json()
    except requests.exceptions.HTTPError as err:
        st.error(f"Error HTTP al cargar sensores: {err}")
        st.error(f"Respuesta del servidor: {err.response.text}")
        if err.response.status_code == 401:
            st.warning("Error 401: No autorizado. Aunque hemos corregido el encabezado a 'ApiKey', el token sigue siendo inválido. ¿Está la clave bien copiada en secrets.toml?")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar sensores: {e}")
        return None

@st.cache_data(ttl=60) # Cachear las lecturas por 1 minuto
def get_last_measurements(headers):
    """Obtiene las últimas mediciones de todos los sensores."""
    try:
        url = f"{API_BASE_URL}/measurements/last"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        st.error(f"Error HTTP al cargar mediciones: {err}")
        st.error(f"Respuesta del servidor: {err.response.text}")
        if err.response.status_code == 401:
            st.warning("Error 401: No autorizado. ¿Está la clave bien copiada en secrets.toml?")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar mediciones: {e}")
        return None

# --- Lógica Principal de la App (sin cambios) ---

st.header("1. Cargar Lista de Sensores")
st.write("Presiona este botón para obtener la lista de todos tus sensores.")

if st.button("Cargar Sensores (GET /sensors)"):
    sensors_data = load_aranet_sensors(API_HEADERS)
    
    if sensors_data:
        if 'sensors' in sensors_data:
            df_sensors = pd.json_normalize(sensors_data['sensors'])
            st.success(f"¡Éxito! Se encontraron {len(df_sensors)} sensores.")
            st.dataframe(df_sensors[['id', 'name', 'type', 'value', 'rssi', 'battery']])
            st.caption("Datos JSON completos:")
            st.json(sensors_data)
        else:
            st.warning("La respuesta no tiene la clave 'sensors'. Mostrando JSON crudo:")
            st.json(sensors_data)
            
st.divider()

st.header("2. Cargar Últimas Mediciones")
st.write("Presiona este botón para obtener la última lectura de todos los sensores.")

if st.button("Cargar Mediciones (GET /measurements/last)"):
    measurements_data = get_last_measurements(API_HEADERS)
    
    if measurements_data:
        if 'data' in measurements_data:
            df_measure = pd.json_normalize(measurements_data['data'])
            st.success(f"¡Éxito! Se encontraron {len(df_measure)} mediciones.")
            st.dataframe(df_measure)
            st.caption("Datos JSON completos:")
            st.json(measurements_data)
        else:
            st.warning("La respuesta no tiene la clave 'data'. Mostrando JSON crudo:")
            st.json(measurements_data)
