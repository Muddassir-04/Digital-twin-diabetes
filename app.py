# ------------------------
# Diabetes Digital Twin Simulator
# ------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------
# FUNCTIONS
# ------------------------

def train_model(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model

def predict_diabetes_risk(patient_df, model):
    """Predict diabetes risk probability"""
    return model.predict_proba(patient_df)[0][1]

def simulate_month(digital_twin, lifestyle="no_change"):
    """Simulate one month of disease progression"""
    twin = digital_twin.copy()
    if lifestyle == "no_change":
        twin["Glucose"] += np.random.normal(5, 2)
        twin["BMI"] += np.random.normal(0.2, 0.1)
    else:  # improved lifestyle
        twin["Glucose"] -= np.random.normal(3, 1)
        twin["BMI"] -= np.random.normal(0.2, 0.05)
    return twin

def run_simulation(patient_df, model, months=12):
    """Run both scenarios (no-change and improved)"""
    
    # No lifestyle change
    digital_twin = patient_df.copy()
    no_change_risk = []
    twin = digital_twin.copy()
    for _ in range(months):
        twin = simulate_month(twin, lifestyle="no_change")
        no_change_risk.append(predict_diabetes_risk(twin, model))
    
    # Improved lifestyle
    digital_twin = patient_df.copy()
    improved_risk = []
    twin = digital_twin.copy()
    for _ in range(months):
        twin = simulate_month(twin, lifestyle="improved")
        improved_risk.append(predict_diabetes_risk(twin, model))
    
    return no_change_risk, improved_risk

def interpret_risk(risk):
    if risk < 0.3:
        return "Low Risk"
    elif risk < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"

def plot_risk(no_change_risk, improved_risk):
    months = range(1, len(no_change_risk) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(months, no_change_risk, marker='o', label="No Lifestyle Change")
    plt.plot(months, improved_risk, marker='o', label="Improved Lifestyle")
    plt.xlabel("Month")
    plt.ylabel("Diabetes Risk Probability")
    plt.title("Digital Twin: Diabetes Risk Projection")
    plt.ylim(0,1)
    plt.legend()
    st.pyplot(plt)

# ------------------------
# STREAMLIT APP LAYOUT
# ------------------------

st.set_page_config(page_title="Diabetes Digital Twin", layout="centered")
st.title("🧬 Diabetes Digital Twin Simulator")
st.write("Simulate diabetes risk over 12 months with lifestyle interventions.")

# ------------------------
# LOAD DATA & TRAIN MODEL
# ------------------------
# Make sure your dataset.csv is in the same folder as this app
df = pd.read_csv("dataset.csv")
model = train_model(df)

# ------------------------
# PATIENT INPUT
# ------------------------
st.sidebar.header("Patient Details")
patient = {
    "Pregnancies": st.sidebar.number_input("Pregnancies", 0, 20, 2),
    "Glucose": st.sidebar.slider("Glucose", 70, 200, 150),
    "BloodPressure": st.sidebar.slider("Blood Pressure", 50, 120, 80),
    "SkinThickness": st.sidebar.slider("Skin Thickness", 0, 60, 30),
    "Insulin": st.sidebar.slider("Insulin", 0, 300, 120),
    "BMI": st.sidebar.slider("BMI", 15.0, 45.0, 32.0),
    "DiabetesPedigreeFunction": st.sidebar.slider("Pedigree Function", 0.1, 2.5, 0.6),
    "Age": st.sidebar.slider("Age", 18, 90, 45)
}

patient_df = pd.DataFrame([patient])

# ------------------------
# BASELINE RISK
# ------------------------
baseline_risk = predict_diabetes_risk(patient_df, model)
st.subheader("📊 Baseline Prediction")
st.write(f"**Diabetes Risk:** {baseline_risk:.2f}")
st.write(f"**Risk Category:** {interpret_risk(baseline_risk)}")

# ------------------------
# 12-MONTH SIMULATION
# ------------------------
no_change_risk, improved_risk = run_simulation(patient_df, model)

st.subheader("📈 12-Month Risk Simulation")
plot_risk(no_change_risk, improved_risk)

# ------------------------
# FINAL MONTH INTERPRETATION
# ------------------------
st.write("**Final Month Risk Interpretation:**")
st.write(f"No Lifestyle Change: {interpret_risk(no_change_risk[-1])}")
st.write(f"Improved Lifestyle: {interpret_risk(improved_risk[-1])}")

st.success("Simulation completed successfully ✅")
