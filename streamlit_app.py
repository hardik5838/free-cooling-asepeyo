## app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Indore Solar Radiation Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS (Indore Context) ---
LATITUDE = 22.71  # North [cite: 5, 24]
I_SC = 1367       # Solar Constant
TAU_ATM = 0.7     # Transmittance approx

# --- SOLAR CALCULATIONS ---
def get_declination(day_of_year):
    """Approximate declination angle delta in degrees."""
    return 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))

@st.cache_data
def calculate_radiation_data(day_of_year, tilt, surface_azimuth):
    """
    Calculates hourly radiation components.
    surface_azimuth: 0=South, -90=East, 90=West, 180=North [cite: 24, 75, 76]
    """
    declination = get_declination(day_of_year)
    delta_rad = np.deg2rad(declination)
    phi_rad = np.deg2rad(LATITUDE)
    beta_rad = np.deg2rad(tilt)
    gamma_rad = np.deg2rad(surface_azimuth)
    
    hours = np.linspace(5, 19, 100) # 5 AM to 7 PM
    results = []
    
    for h in hours:
        # Hour angle (omega): 0 at noon
        omega = (h - 12) * 15
        omega_rad = np.deg2rad(omega)
        
        # 1. Zenith Angle (theta_z)
        cos_theta_z = (np.sin(phi_rad) * np.sin(delta_rad) + 
                       np.cos(phi_rad) * np.cos(delta_rad) * np.cos(omega_rad))
        
        if cos_theta_z <= 0:
            results.append({"Hour": h, "Total": 0, "Beam": 0, "Diffuse": 0})
            continue
            
        # 2. Incidence Angle (theta)
        cos_theta = (np.sin(delta_rad) * np.sin(phi_rad) * np.cos(beta_rad) - 
                     np.sin(delta_rad) * np.cos(phi_rad) * np.sin(beta_rad) * np.cos(gamma_rad) + 
                     np.cos(delta_rad) * np.cos(phi_rad) * np.cos(beta_rad) * np.cos(omega_rad) +
                     np.cos(delta_rad) * np.sin(phi_rad) * np.sin(beta_rad) * np.cos(gamma_rad) * np.cos(omega_rad) +
                     np.cos(delta_rad) * np.sin(beta_rad) * np.sin(gamma_rad) * np.sin(omega_rad))

        # 3. Beam Radiation
        m = 1 / (cos_theta_z + 0.05)
        I_b_normal = I_SC * (TAU_ATM ** m)
        I_b_surface = max(0, I_b_normal * cos_theta)
            
        # 4. Diffuse & Reflected (Simplified model)
        I_d_surface = (I_SC * 0.15 * cos_theta_z) * ((1 + np.cos(beta_rad)) / 2)
        I_r_surface = (I_SC * 0.15 * cos_theta_z) * 0.2 * ((1 - np.cos(beta_rad)) / 2)
        
        diffuse_total = max(0, I_d_surface + I_r_surface)
        
        results.append({
            "Hour": h, 
            "Total": I_b_surface + diffuse_total, 
            "Beam": I_b_surface, 
            "Diffuse": diffuse_total
        })

    return pd.DataFrame(results)

# --- USER INTERFACE ---
st.title("☀️ Solar Radiation Simulation: Indore (22.7°N)")
st.markdown("""
This application analyzes how surface slope and orientation affect solar energy gains. 
In tropical latitudes like Indore, optimal summer strategies require nearly horizontal surfaces. [cite: 8]
""")

with st.sidebar:
    st.header("Simulation Parameters")
    season = st.radio("Select Season", ["Summer Solstice", "Winter Solstice"])
    day_val = 172 if season == "Summer Solstice" else 355
    
    st.divider()
    st.subheader("Surface Configuration")
    
    # Preset Slopes
    slope_type = st.selectbox("Slope Preset", ["Optimum", "Vertical", "Custom"])
    if slope_type == "Optimum":
        # Logic: |Lat - Declination| [cite: 25]
        tilt = 0.75 if season == "Summer Solstice" else 46.15
        st.info(f"Calculated Optimum: {tilt}°")
    elif slope_type == "Vertical":
        tilt = 90.0
    else:
        tilt = st.slider("Custom Tilt (degrees)", 0.0, 90.0, 22.7)

    # Orientation
    orient_name = st.selectbox("Orientation", ["South", "East", "West", "North", "Custom"])
    orient_map = {"South": 0, "East": -90, "West": 90, "North": 180}
    if orient_name == "Custom":
        azimuth = st.slider("Azimuth (South=0, West=90)", -180, 180, 0)
    else:
        azimuth = orient_map[orient_name]

# --- DATA PROCESSING & PLOTTING ---
df = calculate_radiation_data(day_val, tilt, azimuth)

col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Total"], name="Total Radiation", fill='tozeroy', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Beam"], name="Beam Component", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Diffuse"], name="Diffuse/Reflected", line=dict(dash='dot')))

    fig.update_layout(
        title=f"Radiation Profile: {season} at {tilt}° Tilt ({orient_name})",
        xaxis_title="Solar Time (h)",
        yaxis_title="Irradiance (W/m²)",
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(range=[0, 1200])
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Peak Irradiance", f"{df['Total'].max():.2f} W/m²")
    st.metric("Total Daily Energy", f"{(df['Total'].sum() * (14/100)):.2f} Wh/m²")
    
    st.write("---")
    st.write("**Quick Comments:**")
    if tilt == 90 and season == "Summer Solstice" and orient_name == "South":
        st.warning("Note the 'dip' at noon! In Indore, the summer sun passes North of the zenith, leaving South vertical walls in shade. ")
    elif tilt < 5:
        st.success("Flat surfaces are nearly optimal for Indore summers. [cite: 30]")

st.divider()
st.subheader("Assignment Reference Data")
st.table(pd.DataFrame({
    "Scenario": ["Summer Opt", "Winter Opt", "Vertical South (Summer)"],
    "Slope": ["0.75°", "46.15°", "90°"],
    "Insight": ["Essentially Horizontal [cite: 30]", "Maximize low sun gains [cite: 394]", "Midday dip due to sun orientation "]
}))
