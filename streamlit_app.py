## app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Indore Report Generator", layout="centered")

# --- CONSTANTS (Indore Context) ---
LATITUDE = 22.7  # North [cite: 5]
I_SC = 1367       
TAU_ATM = 0.7     

# --- SOLAR MATH ---
def get_declination(day_of_year):
    return 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))

def calculate_radiation(day_of_year, tilt, azimuth):
    declination = get_declination(day_of_year)
    delta_rad, phi_rad = np.deg2rad(declination), np.deg2rad(LATITUDE)
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

# --- REPORT CONTENT MAPPING ---
# Mapping figures to their specific settings and descriptions from your assignment
figures = {
    "Figure 1": {"season": "Summer", "slope": 0.75, "orient": 0, "desc": "Summer radiation at optimum slope (Horizontal). Peaking high as sun passes near zenith. [cite: 53, 54]"},
    "Figure 2": {"season": "Winter", "slope": 46.15, "orient": 0, "desc": "Winter radiation at optimum slope (46.15°). Strong capture of low southern sun. [cite: 65, 66]"},
    "Figure 3": {"season": "Summer", "slope": 0.75, "orient": -90, "desc": "Summer solstice radiation facing East. [cite: 101]"},
    "Figure 4": {"season": "Winter", "slope": 46.15, "orient": -90, "desc": "Winter solstice radiation facing East. [cite: 124]"},
    "Figure 7": {"season": "Summer", "slope": 0.75, "orient": 180, "desc": "Summer solstice radiation facing North. [cite: 182]"},
    "Figure 9": {"season": "Summer", "slope": 90, "orient": 0, "desc": "Summer South Vertical: Sun passes North of zenith, leading to midday self-shading. [cite: 242]"},
    "Figure 10": {"season": "Winter", "slope": 90, "orient": 0, "desc": "Winter South Vertical: Good incidence angle for vertical walls in winter. [cite: 260]"}
}

# --- UI ---
st.title("📄 Report Generator: Solar Radiation")
st.sidebar.header("Selection for Print")
selected_fig = st.sidebar.selectbox("Select Figure for Report", list(figures.keys()))

config = figures[selected_fig]
day_val = 172 if config["season"] == "Summer" else 355
df = calculate_radiation(day_val, config["slope"], config["orient"])

# Chart Styling
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Hour"], y=df["It"], name="It (Total)", line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=df["Hour"], y=df["Itb"], name="Itb (Beam)", line=dict(dash='dash', color='blue')))
fig.add_trace(go.Scatter(x=df["Hour"], y=df["Itd"], name="Itd (Diffuse)", line=dict(dash='dot', color='green')))

fig.update_layout(
    title=f"{selected_fig}: {config['season']} Solstice (Slope: {config['slope']}°, Orient: {config['orient']}°)",
    xaxis_title="Solar Time (h)",
    yaxis_title="Radiation (W/m²)",
    template="plotly_white",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Formal Report Text Box
st.markdown(f"**Description:** {config['desc']}")
st.info("💡 **Report Tip:** Use the 'Download as PNG' button on the top right of the chart to save this image for your PDF.")
