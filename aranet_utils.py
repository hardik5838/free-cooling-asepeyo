## aranet_utils.py
import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "https://aranet.cloud/api/v1"

@st.cache_data(ttl=600) # Cachear por 10 minutos
def get_api_headers():
    """
    Carga la clave de API desde los secretos y devuelve el 
    diccionario de encabezados (headers) correcto.
    """
    try:
        api_key = st.secrets["ARANET_API_KEY"]
        headers = {
            "ApiKey": api_key,
            "Accept": "application/json"
        }
        return headers
    except KeyError:
        st.error("Error: 'ARANET_API_KEY' no encontrada en .streamlit/secrets.toml")
        return None
    except Exception as e:
        st.error(f"Error al configurar headers: {e}")
        return None

@st.cache_data(ttl=600) # Cachear la lista de sensores por 10 minutos
def load_sensors():
    """
    Carga la lista de sensores desde la API de Aranet.
    Devuelve un diccionario para mapear Nombre -> ID.
    """
    headers = get_api_headers()
    if not headers:
        return {}

    try:
        url = f"{API_BASE_URL}/sensors"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Procesamos la lista de sensores del JSON
        if 'sensors' in data:
            sensor_name_map = {}
            for sensor in data['sensors']:
                name = sensor.get('name', 'Sensor Sin Nombre')
                id_ = sensor.get('id')
                if id_:
                    # Añadimos el ID al nombre para evitar duplicados
                    sensor_name_map[f"{name} (id: {id_})"] = id_
            return sensor_name_map
        else:
            st.error("Respuesta de API (sensores) no contiene la clave 'sensors'.")
            return {}
            
    except requests.exceptions.HTTPError as err:
        st.error(f"Error HTTP al cargar sensores: {err}")
        return {}
    except Exception as e:
        st.error(f"Error inesperado al cargar sensores: {e}")
        return {}

@st.cache_data(ttl=60) # Cachear las mediciones por 1 minuto
def get_processed_measurements():
    """
    Carga las últimas mediciones y las procesa de formato 'largo' a 'ancho'.
    Devuelve un diccionario: { "sensor_id": {"temperature": T, "humidity": H}, ... }
    """
    headers = get_api_headers()
    if not headers:
        return {}

    try:
        url = f"{API_BASE_URL}/measurements/last"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # El JSON usa la clave 'readings', no 'data'
        if 'readings' not in data:
            st.error("Respuesta de API (mediciones) no contiene la clave 'readings'.")
            return {}
            
        processed_data = {}
        
        # Iteramos sobre la lista 'readings'
        for reading in data['readings']:
            sensor_id = reading.get('sensor')
            metric = reading.get('metric')
            value = reading.get('value')
            
            if not sensor_id:
                continue
                
            # Si es la primera vez que vemos este sensor, creamos su entrada
            if sensor_id not in processed_data:
                processed_data[sensor_id] = {}
            
            # Asignamos temperatura (metric "1") y humedad (metric "2")
            if metric == "1":
                processed_data[sensor_id]["temperature"] = value
            elif metric == "2":
                processed_data[sensor_id]["humidity"] = value
                
        return processed_data

    except requests.exceptions.HTTPError as err:
        st.error(f"Error HTTP al cargar mediciones: {err}")
        return {}
    except Exception as e:
        st.error(f"Error inesperado al cargar mediciones: {e}")
        return {}
