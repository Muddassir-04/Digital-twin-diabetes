# 🧬 Diabetes Digital Twin Simulator

## Project Overview
This project implements a **Digital Twin for Diabetes** using **Python, Machine Learning, and Streamlit**.  
A Digital Twin is a **virtual representation of a patient** that allows simulation of disease progression and evaluation of interventions over time.  

Using real healthcare data, this system predicts a patient’s **baseline diabetes risk**, simulates **12-month risk progression**, and compares two scenarios:  

1. **No Lifestyle Change** – the patient continues current habits  
2. **Improved Lifestyle** – the patient adopts healthier habits  

The system also provides **clinical interpretations** (Low / Moderate / High Risk) for each scenario.

---

## Features
- ✅ Predict baseline diabetes risk using Logistic Regression  
- ✅ Create a **Digital Twin** of the patient  
- ✅ Simulate **monthly disease progression** over 12 months  
- ✅ Compare **no-change vs improved lifestyle scenarios**  
- ✅ Visualize risk progression with interactive plots  
- ✅ Provide clinical interpretation of risk  
- ✅ Fully interactive **Streamlit app** for real-time input and visualization  

---

## Dataset
The project uses the **Pima Indians Diabetes Dataset**, which includes features such as:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  
- Outcome (0 = non-diabetic, 1 = diabetic)  

> Make sure `dataset.csv` is in the same folder as the app.

---

## How It Works

1. **Data Loading & Preprocessing**  
   - Missing values are handled appropriately  
   - Dataset is used to train a Logistic Regression model

2. **Model Training**  
   - Features scaled using `StandardScaler`  
   - Logistic Regression model predicts diabetes risk probability  

3. **Digital Twin Simulation**  
   - Creates a copy of the patient as a virtual twin  
   - Simulates monthly progression over 12 months  
   - Two scenarios: no-change and improved lifestyle  
   - Probabilistic changes applied to Glucose and BMI to reflect realistic variation  

4. **Visualization & Interpretation**  
   - Risk curves plotted over 12 months  
   - Final month risk categorized into Low / Moderate / High  

---

