import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="FleetAfya", layout="centered")

# ---------------------------
# Load pre-trained model + scaler
# ---------------------------
model = joblib.load("soh_xgb_africa.joblib")
scaler = joblib.load("scaler_africa.joblib")

# ---------------------------
# Example Fleet Data
# ---------------------------
df = pd.DataFrame([
    {
        "Serial": "BAT-001",
        "Age_days": 300, 
        "Cycle_count": 400, 
        "Avg_ambient": 28, 
        "Max_pack_temp": 38,
        "Avg_C_rate": 0.6, 
        "Avg_SOC": 65, 
        "Hours_above_40": 20,
        "Cumulative_Wh": 1_200_000,
    },
    {
        "Serial": "BAT-002",
        "Age_days": 1200, 
        "Cycle_count": 1800, 
        "Avg_ambient": 34, 
        "Max_pack_temp": 46,
        "Avg_C_rate": 0.9, 
        "Avg_SOC": 70, 
        "Hours_above_40": 150,
        "Cumulative_Wh": 3_500_000,
    },
    {
        "Serial": "BAT-003",
        "Age_days": 2000, 
        "Cycle_count": 2800, 
        "Avg_ambient": 40, 
        "Max_pack_temp": 55,
        "Avg_C_rate": 1.2, 
        "Avg_SOC": 80, 
        "Hours_above_40": 500,
        "Cumulative_Wh": 11_000_000,
    },
])

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🔋 FleetAfya - Dashboard 🔋</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Track your batteries. Predict their future.</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# STEP 1: Fleet Overview
# ---------------------------
st.markdown("### 📦 Step 1: Battery Fleet Overview")
st.caption("Live Battery telemetry from BMS (predictions happen after you select a pack)")

styled_df = df.style.set_table_styles(
    [
        {'selector': 'thead th', 'props': [('background-color', '#2E86C1'), ('color', 'white'), ('text-align', 'center')]},
        {'selector': 'tbody td', 'props': [('text-align', 'center')]}
    ]
).format(thousands=",")
st.dataframe(styled_df, use_container_width=True)

# ---------------------------
# STEP 2: Select and Predict
# ---------------------------
st.markdown("### 🔎 Step 2: Inspect a Specific Battery Pack")

serial = st.selectbox("Select Battery Serial Number", df["Serial"].unique())
pack = df[df["Serial"] == serial].iloc[0]

# Prepare features for prediction
X_input = np.array([[pack["Age_days"], pack["Cycle_count"], pack["Avg_ambient"], 
                     pack["Max_pack_temp"], pack["Avg_C_rate"], pack["Avg_SOC"], 
                     pack["Hours_above_40"], pack["Cumulative_Wh"]]])
X_scaled = scaler.transform(X_input)
soh_pred = model.predict(X_scaled)[0]

# Cycles left estimation
cycles_left = int((soh_pred / 100) * 1000)

# ---------------------------
# Dynamic Card Colors Function
# ---------------------------
def get_soh_color(soh):
    if soh > 80:
        return "#82E0AA"  # Green
    elif soh > 60:
        return "#F7DC6F"  # Yellow
    else:
        return "#F1948A"  # Red

def get_projection_color(years_left):
    if years_left > 3:
        return "#85C1E9"  # Blue
    elif years_left > 1:
        return "#F7DC6F"  # Yellow
    else:
        return "#F1948A"  # Red

soh_color = get_soh_color(soh_pred)

# ---------------------------
# Step 2: Cards
# ---------------------------
st.markdown(
    f"""
    <div style='display: flex; justify-content: space-around; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;'>
        <div style='background: {soh_color}; padding:25px; border-radius:15px; text-align:center; width:30%; min-width:200px; box-shadow:2px 2px 10px rgba(0,0,0,0.1); color:#1B2631;'>
            <h3 style='margin-bottom:10px;'>Predicted SoH</h3>
            <h2 style='margin:0;'>{soh_pred:.1f}%</h2>
        </div>
        <div style='background: #85C1E9; padding:25px; border-radius:15px; text-align:center; width:30%; min-width:200px; box-shadow:2px 2px 10px rgba(0,0,0,0.1); color:#1B2631;'>
            <h3 style='margin-bottom:10px;'>Cycles Completed</h3>
            <h2 style='margin:0;'>{pack['Cycle_count']}</h2>
        </div>
        <div style='background: #BB8FCE; padding:25px; border-radius:15px; text-align:center; width:30%; min-width:200px; box-shadow:2px 2px 10px rgba(0,0,0,0.1); color:#1B2631;'>
            <h3 style='margin-bottom:10px;'>Estimated Cycles Left</h3>
            <h2 style='margin:0;'>{cycles_left}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True
)

# ---------------------------
# STEP 3: Fleet Usage Inputs
# ---------------------------
st.markdown("### ⚙️ Step 3: Enter Battery Usage Info")
col4, col5 = st.columns(2)
with col4:
    charges_per_day = st.number_input("🔌 Charges per day", 0.5, 3.0, 1.2, step=0.1)
with col5:
    avg_range_km = st.number_input("🛣️ Avg range per full charge (km)", 50, 500, 300, step=10)

# Projected outcomes
time_left_days = cycles_left / charges_per_day if charges_per_day > 0 else 0
time_left_years = time_left_days / 365
distance_left = cycles_left * avg_range_km

st.divider()

# Step 4: Dynamic Projected Outcomes Card
proj_color = get_projection_color(time_left_years)
st.markdown(
    f"""
    <div style='background:{proj_color}; padding:25px; border-radius:15px; box-shadow:2px 2px 10px rgba(0,0,0,0.1); color:#1B2631; text-align:center;'>
        <h3>📊 Projected Battery Outcomes</h3>
        <p style='margin:5px 0;'><b>Battery Time Left:</b> ~{time_left_years:.1f} years</p>
        <p style='margin:5px 0;'><b>Battery Distance Left:</b> ~{distance_left:,.0f} km</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# RECOMMENDATION
# ---------------------------
if soh_pred > 80:
    st.success("✅ Battery health is good — suitable for heavy fleet duty.")
elif soh_pred > 60:
    st.warning("⚠️ Battery moderately degraded — plan replacements soon.")
else:
    st.error("❌ Battery poor — consider replacement or 2nd-life use.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Powered by FleetAfya's battery data + ML model ⚡<br>Data tuned for African EV fleet context 🌍</p>", unsafe_allow_html=True)
