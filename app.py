import streamlit as st
import numpy as np
import joblib

# Load models
regressor = joblib.load("regressor.pkl")
classifier = joblib.load("classifier.pkl")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800;900&display=swap" rel="stylesheet">

<style>
.glassy-title {
    font-family: 'Orbitron', sans-serif;
    text-align: center;
    font-size: 75px;
    font-weight: 900;
    background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 4px;
    text-shadow: 0 0 25px rgba(0, 150, 255, 0.8);
    margin-bottom: 10px;
}

.glassy-subtitle {
    font-family: 'Orbitron', sans-serif;
    text-align: center;
    font-size: 26px;
    font-weight: 600;
    color: #ff3333;  /* keep subtitle red if you want contrast */
    letter-spacing: 2px;
    margin-top: -10px;
}
</style>

<div class="glassy-title">
    Glassy Tech
</div>

<div class="glassy-subtitle">
    SPS Spinel ML Prediction System
</div>
""", unsafe_allow_html=True)
st.image("Spinel.jpeg",
         caption="AI-Based Defect Detection in Spinel Oxide Ceramics",
         use_container_width=True)
st.markdown(
    "<h2 style='font-weight: 800;'>Enter SPS Processing Parameters</h2>",
    unsafe_allow_html=True
)

# User Inputs
temperature = st.number_input("Temperature (°C)", 500.0, 1750.0, 1300.0)
pressure = st.number_input("Pressure (MPa)", 20.28, 150.0, 50.0)
heating_rate = st.number_input("Heating Rate (°C/min)", 50.80, 600.0, 150.0)
holding_time = st.number_input("Holding Time (min)", 5.00, 90.0, 10.0)
particle_size = st.number_input("Particle Size (µm)", 50.29, 100.0, 75.0)
vacancy_conc = st.number_input("Vacancy Concentration", 1.00e-05, 1.00e-02, 1.00e-04)
diffusion_coeff = st.number_input("Diffusion Coefficient (m²/s)", 1.00e-12, 1.00e-09, 1.00e-11)

if st.button("Predict"):

    input_data = np.array([[temperature, pressure, heating_rate,
                            holding_time, particle_size,
                            vacancy_conc, diffusion_coeff]])

    # Regression Prediction
    reg_prediction = regressor.predict(input_data)

    # Classification Prediction
    clf_prediction = classifier.predict(input_data)
    defect_prob = classifier.predict_proba(input_data)[0][1]

    st.subheader("Predicted Material Properties")

    st.write(f"Final Density: {reg_prediction[0][0]:.4f}")
    st.write(f"Porosity: {reg_prediction[0][1]:.4f}")
    st.write(f"Grain Size: {reg_prediction[0][2]:.4f}")
    st.write(f"Hardness: {reg_prediction[0][3]:.4f}")

    st.subheader("Defect Prediction")

    if clf_prediction[0] == 1:
        st.error(f"Defect Likely (Probability: {defect_prob:.2f})")
    else:
        st.success(f"No Defect (Probability: {1-defect_prob:.2f})")


st.markdown("""
<style>
.team-box {
    background-color: rgba(0, 123, 255, 0.08);
    padding: 22px;
    border-radius: 12px;
    border: 1px solid rgba(0, 123, 255, 0.2);
    margin-top: 30px;
}

.team-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 15px;
}

.team-members {
    font-size: 18px;
    line-height: 2;
    margin-left: 30px;
}
</style>

<div class="team-box">
    <div class="team-title">Team Members</div>
    <div class="team-members">
        Alphina Seles L <br>
        Arjun M Rao <br>
        Krishnakumar V
    </div>
</div>
""", unsafe_allow_html=True)
