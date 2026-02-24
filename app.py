import streamlit as st
import numpy as np
import joblib

# Load models
regressor = joblib.load("regressor.pkl")
classifier = joblib.load("classifier.pkl")

st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4;'>
        Glassy Tech
    </h1>
    <h4 style='text-align: corner;font-weight: bold;'>
        SPS Spinel ML Prediction System
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
st.image("Spinel.jpeg",
         caption="AI-Based Defect Detection in Spinel Oxide Ceramics",
         use_container_width=True)

st.write("Enter SPS Processing Parameters")

# User Inputs
temperature = st.number_input("Temperature (°C)", 800.0, 2000.0, 1200.0)
pressure = st.number_input("Pressure (MPa)", 10.0, 100.0, 50.0)
heating_rate = st.number_input("Heating Rate (°C/min)", 1.0, 100.0, 10.0)
holding_time = st.number_input("Holding Time (min)", 1.0, 60.0, 10.0)
particle_size = st.number_input("Particle Size (µm)", 0.1, 10.0, 2.0)
vacancy_conc = st.number_input("Vacancy Concentration", 0.0, 1.0, 0.01)
diffusion_coeff = st.number_input("Diffusion Coefficient", 0.0, 1.0, 0.001)

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


st.markdown(
    """
    <div style='text-align: corner;'>
        <h3>Team Members</h3>
        <p>
        Alphina Seles L<br>
        Arjun M Rao <br>
        Krishnakumar V <br>
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)
