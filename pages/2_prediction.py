import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as componenets

st.set_page_config(initial_sidebar_state='expanded')

with open('best_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)


categorical_column =  ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane','bp']
numerical_column = ['age', 'sg', 'hemo', 'al', 'bgr', 'bu', 'sc', 'sod', 'pot', 'wbcc', 'rbcc', 'pcv', 'su']

feature_names = categorical_column + numerical_column

kidney = pd.read_csv('chronic_kidney_disease.csv')
x = kidney[feature_names]
y = kidney['class']

metadata = {
    "age_min": 0, "age_max": 100,
    "sg_min": 1.005, "sg_max": 1.030,
    "hemo_min": 5.0, "hemo_max": 17.0,
    "al_min": 0, "al_max": 5,
    "bgr_min": 50, "bgr_max": 500,
    "bu_min": 10, "bu_max": 200,
    "sc_min": 0.4, "sc_max": 15.0,
    "sod_min": 100, "sod_max": 160,
    "pot_min": 2.0, "pot_max": 10.0,
    "wbcc_min": 2000, "wbcc_max": 18000,
    "rbcc_min": 2.5, "rbcc_max": 7.5,
    "pcv_min": 20, "pcv_max": 55,
    "su_min": 0, "su_max": 5,
    "rbc_categories": ["abnormal", "normal"],
    "pc_categories": ["abnormal", "normal"],
    "pcc_categories": ["not present", "present"],
    "ba_categories": ["not present", "present"],
    "htn_categories": ["no", "yes"],
    "dm_categories": ["no", "yes"],
    "cad_categories": ["no", "yes"],
    "appet_categories": ["poor", "good"],
    "pe_categories": ["no", "yes"],
    "ane_categories": ["no", "yes"],
    "bp_categories": ["no", "yes"]
}

class_names = np.unique(kidney['class']).tolist()

user_input = []
st.sidebar.header("Enter Patient Data")

for col in categorical_column:
    col_value = st.sidebar.radio(f"Select {col}",
                                 metadata[f"{col}_categories"])
    encoded_value = metadata[f"{col}_categories"].index(col_value) 
    user_input.append(encoded_value)

step_map = {
    'age': 1, 'sg': 0.001, 'hemo': 0.1, 'al': 1, 'bgr': 5, 'bu': 5, 'sc': 0.1,
    'sod': 1, 'pot': 0.1, 'wbcc': 100, 'rbcc': 0.1, 'pcv': 1, 'su': 1
}

for col in numerical_column:
    num_value = st.sidebar.number_input(
        f"Select {col}",
        min_value=float(metadata[f"{col}_min"]),
        max_value=float(metadata[f"{col}_max"]),
        step=float(step_map.get(col, 1)))
    user_input.append(num_value)

user_df = pd.DataFrame([user_input], columns=feature_names)


st.write("### Processed Input Data for Prediction:")
st.dataframe(user_df)

if st.button("Predict"):
    prediction = model.predict(user_df)[0]
    prediction_proba = model.predict_proba(user_df)[:, 1][0]

    if prediction == 1:
        st.error(f"🔴 The model predicts **Chronic Kidney Disease (CKD)** with a probability of {prediction_proba:.2f}")
    else:
        st.success(f"🟢 The model predicts **No CKD** with a probability of {1 - prediction_proba:.2f}")

st.subheader("Feature Meanings")

feature_info = """
### Numerical Features:
- **Age**: Age in years  
- **Blood Pressure (bp)**: Measured in mm/Hg  
- **Blood Glucose Random (bgr)**: Measured in mg/dl  
- **Blood Urea (bu)**: Measured in mg/dl  
- **Serum Creatinine (sc)**: Measured in mg/dl  
- **Sodium (sod)**: Measured in mEq/L  
- **Potassium (pot)**: Measured in mEq/L  
- **Hemoglobin (hemo)**: Measured in gms  
- **Packed Cell Volume (pcv)**  
- **White Blood Cell Count (wc)**: Measured in cells/cumm  
- **Red Blood Cell Count (rc)**: Measured in millions/cmm  

### Nominal Features:
- **Specific Gravity (sg)**: (1.005,1.010,1.015,1.020,1.025)  
- **Albumin (al)**: (0,1,2,3,4,5)  
- **Sugar (su)**: (0,1,2,3,4,5)  
- **Red Blood Cells (rbc)**: (normal, abnormal)  
- **Pus Cell (pc)**: (normal, abnormal)  
- **Pus Cell Clumps (pcc)**: (present, not present)  
- **Bacteria (ba)**: (present, not present)  
- **Hypertension (htn)**: (yes, no)  
- **Diabetes Mellitus (dm)**: (yes, no)  
- **Coronary Artery Disease (cad)**: (yes, no)  
- **Appetite (appet)**: (good, poor)  
- **Pedal Edema (pe)**: (yes, no)  
- **Anemia (ane)**: (yes, no)  
"""

with st.expander("ℹ️ Click here to see feature meanings"):
    st.markdown(feature_info)