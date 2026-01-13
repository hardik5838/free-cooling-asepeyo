## app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Indore Solar Report Generator", layout="centered")

# --- CONSTANTS (Indore Context) ---
LAT_INDORE = 22.7  # [cite: 5, 24]
I_SC = 1367        
TAU_ATM = 0.7     

# --- SOLAR CALCULATIONS ---
def get_declination(day_of_year):
    return 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))

def calculate_radiation(day_of_year, tilt, azimuth):
    declination = get_declination(day_of_year)
    delta_rad, phi_rad = np.deg2rad(declination), np.deg2rad(LAT_INDORE)
    beta_rad, gamma_rad = np.deg2rad(tilt), np.deg2rad(azimuth)
    
    hours = np.linspace(5, 19, 100)
    data = []
    for h in hours:
        omega_rad = np.deg2rad((h - 12) * 15)
        cos_theta_z = (np.sin(phi_rad) * np.sin(delta_rad) + 
                       np.cos(phi_rad) * np.cos(delta_rad) * np.cos(omega_rad))
        
        if cos_theta_z <= 0:
            data.append({"Hour": h, "It": 0, "Itb": 0, "Itd": 0})
            continue
            
        cos_theta = (np.sin(delta_rad) * np.sin(phi_rad) * np.cos(beta_rad) - 
                     np.sin(delta_rad) * np.cos(phi_rad) * np.sin(beta_rad) * np.cos(gamma_rad) + 
                     np.cos(delta_rad) * np.cos(phi_rad) * np.cos(beta_rad) * np.cos(omega_rad) +
                     np.cos(delta_rad) * np.sin(phi_rad) * np.sin(beta_rad) * np.cos(gamma_rad) * np.cos(omega_rad) +
                     np.cos(delta_rad) * np.sin(beta_rad) * np.sin(gamma_rad) * np.sin(omega_rad))

        m = 1 / (cos_theta_z + 0.05)
        I_b_normal = I_SC * (TAU_ATM ** m)
        Itb = max(0, I_b_normal * cos_theta)
        Itd = (I_SC * 0.15 * cos_theta_z) * ((1 + np.cos(beta_rad)) / 2)
        data.append({"Hour": h, "It": Itb + Itd, "Itb": Itb, "Itd": Itd})
    return pd.DataFrame(data)

# --- COMPLETE FIGURE MAPPING ---
# Based on your requirements and assignment logic [cite: 73, 207, 394]
figures = {
    "Figure 5": {"season": "Summer", "slope": 0.75, "orient": 90, "desc": "Summer solstice radiation facing West at nearly horizontal slope (0.75°). [cite: 140]"},
    "Figure 6": {"season": "Winter", "slope": 46.15, "orient": 90, "desc": "Winter solstice radiation facing West at optimum slope (46.15°). Shows afternoon peak. "},
    "Figure 8": {"season": "Winter", "slope": 46.15, "orient": 180, "desc": "Winter solstice radiation facing North. Low radiation as sun is in the South sky. [cite: 77, 206]"},
    "Figure 11": {"season": "Summer", "slope": 90, "orient": -90, "desc": "Summer solstice radiation on a Vertical East surface. Peak in early morning. [cite: 214, 281]"},
    "Figure 12": {"season": "Winter", "slope": 90, "orient": -90, "desc": "Winter solstice radiation on a Vertical East surface. [cite: 304]"},
    "Figure 13": {"season": "Summer", "slope": 90, "orient": 90, "desc": "Summer solstice radiation on a Vertical West surface. Peak in late afternoon. [cite: 214, 324]"},
    "Figure 14": {"season": "Winter", "slope": 90, "orient": 90, "desc": "Winter solstice radiation on a Vertical West surface. [cite: 344]"},
    "Figure 15": {"season": "Summer", "slope": 90, "orient": 180, "desc": "Summer Vertical North: Receives beam radiation at noon due to Indore's latitude. [cite: 213, 368]"},
    "Figure 16": {"season": "Winter", "slope": 90, "orient": 180, "desc": "Winter Vertical North: Completely shaded from beam radiation; only diffuse light. [cite: 216, 387]"}
}

# --- UI ---
st.title("☀️ Indore Solar Assignment: Remaining Figures")
st.write("Use this tool to generate the final set of graphs for your report.")

selected_fig = st.sidebar.selectbox("Select Missing Figure", list(figures.keys()))

config = figures[selected_fig]
day_val = 172 if config["season"] == "Summer" else 355
df = calculate_radiation(day_val, config["slope"], config["orient"])

# Chart Styling
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Hour"], y=df["It"], name="It (Total)", line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=df["Hour"], y=df["Itb"], name="Itb (Beam)", line=dict(dash='dash', color='blue')))
fig.add_trace(go.Scatter(x=df["Hour"], y=df["Itd"], name="Itd (Diffuse)", line=dict(dash='dot', color='green')))

fig.update_layout(
    title=f"<b>{selected_fig}</b>: {config['season']} (Slope {config['slope']}°, Orient {config['orient']}°)",
    xaxis_title="Solar Time (h)",
    yaxis_title="Irradiance (W/m²)",
    template="plotly_white",
    yaxis=dict(range=[0, 1100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Formal Report Text Box
st.markdown(f"**Report Description:** {config['desc']}")
st.success(f"Parameters applied: Latitude {LAT_INDORE}°N, Day {day_val}, Slope {config['slope']}°, Azimuth {config['orient']}°.")
