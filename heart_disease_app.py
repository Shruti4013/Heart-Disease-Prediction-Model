import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 🎯 Load and preprocess dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\sahil\Downloads\heart.csv")
        
        # Print original shape for debugging
        st.write(f"Original data shape: {df.shape}")
        
        # Convert all categorical columns to numeric (case-insensitive)
        sex_map = {'male': 1, 'female': 0}
        cp_map = {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}
        fbs_map = {'true': 1, 'false': 0, '1': 1, '0': 0}
        restecg_map = {'normal': 0, 'st-t wave abnormality': 1, 'left ventricular hypertrophy': 2}
        exang_map = {'yes': 1, 'no': 0}
        slope_map = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
        thal_map = {'normal': 1, 'fixed defect': 2, 'reversible defect': 3}

        # Convert string columns to lowercase and map
        if "sex" in df.columns and df["sex"].dtype == object:
            df["sex"] = df["sex"].astype(str).str.lower().map(sex_map).fillna(1)
        
        if "cp" in df.columns and df["cp"].dtype == object:
            df["cp"] = df["cp"].astype(str).str.lower().map(cp_map).fillna(0)
        
        # Convert remaining categorical columns
        categorical_cols = {
            'fbs': fbs_map,
            'restecg': restecg_map,
            'exang': exang_map,
            'slope': slope_map,
            'thal': thal_map
        }
        
        for col, mapping in categorical_cols.items():
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.lower().map(mapping).fillna(0)
        
        # Ensure all columns are numeric (without dropping)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop only rows where target is missing
        if 'target' in df.columns:
            df = df.dropna(subset=['target'])
        
        # Verify we still have data
        if df.empty:
            st.error("Error: No data remaining after preprocessing!")
            return None
            
        st.write(f"Processed data shape: {df.shape}")
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# 🎨 Page settings
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤", layout="centered")

st.markdown("<h1 style='text-align: center; color: crimson;'>❤ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter patient information to predict heart disease risk</h4><br>", unsafe_allow_html=True)

# 📦 Load and preprocess data
data = load_data()

if data is None:
    st.stop()  # Stop execution if data loading failed

X = data.drop("target", axis=1)
y = data["target"]

# Verify we have features
if X.shape[0] == 0:
    st.error("Error: No samples available for training!")
    st.stop()

scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
except Exception as e:
    st.error(f"Model training failed: {str(e)}")
    st.stop()

# [Rest of your existing UI code...]
  

# 💡 Input mappings (for the UI)
sex_map = {'Male': 1, 'Female': 0}
cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
fbs_map = {'Yes': 1, 'No': 0}
restecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
exang_map = {'Yes': 1, 'No': 0}
slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

# 🧾 Input form
with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 80, 50)
        sex = st.radio("Sex", list(sex_map.keys()))
        cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
        fbs = st.radio("Fasting Blood Sugar > 120?", list(fbs_map.keys()))
        restecg = st.selectbox("Resting ECG", list(restecg_map.keys()))

    with col2:
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina?", list(exang_map.keys()))
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
        ca = st.slider("Number of Major Vessels", 0, 3, 0)
        thal = st.selectbox("Thalassemia", list(thal_map.keys()))

    submitted = st.form_submit_button("🔍 Predict")

# 🚀 Make prediction
if submitted:
    input_data = np.array([[
        age,
        sex_map[sex],
        cp_map[cp],
        trestbps,
        chol,
        fbs_map[fbs],
        restecg_map[restecg],
        thalach,
        exang_map[exang],
        oldpeak,
        slope_map[slope],
        ca,
        thal_map[thal]
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of heart disease

    st.markdown("---")
    if prediction[0] == 0:
        st.success(f"✅ Prediction: No Heart Disease Detected (Probability: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"⚠ Prediction: High Risk of Heart Disease (Probability: {probability:.2%})")
        st.markdown("<h3 style='color: red;'>🚨 Please consult your doctor immediately!</h3>", unsafe_allow_html=True)