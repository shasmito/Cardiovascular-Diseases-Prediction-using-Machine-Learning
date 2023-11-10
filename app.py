

# -*- coding: utf-8 -*-

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('model/trained_model.pkl', 'rb'))

# Function for prediction
def Cardiovascular_disease_Prediction(input_data):
    # Define mappings for categorical variables
    sex_mapping = {"Male-0": 0, "Female-1": 1}
    cp_mapping = {"Angina Pectoris-0": 0, "Myocardial Infarction (Heart Attack)-1": 1, "Gastroesophageal Reflux Disease-2": 2, "Costochondritis-3": 3}
    fbs_mapping = {"No-0": 0, "Yes-1": 1}
    restecg_mapping = {"ECG Results Regular-0": 0, "ECG Results Severity-1": 1, "ECG Results Indeterminate-2": 2}
    exang_mapping = {"No-0": 0, "Yes-1": 1}
    slope_mapping = {"Upsloping ST Segment-0": 0, "Horizontal ST Segment-1": 1, "Downsloping ST Segment-2": 2}
    ca_mapping = {"No Major Coronary Vessel-0": 0, "One Major Coronary Vessel-1": 1, "Two Major Coronary Vessel-2": 2, "Three  Major Coronary Vessel-3": 3}
    thal_mapping = {"Normal-0": 0, "Musculoskeletal Pain-1": 1, "Pleuritic Pain-2": 2, "Panic Attack-3": 3}

    age, sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = input_data

    input_data_array = [
        float(age),
        sex_mapping[sex],
        cp_mapping[cp],
        float(restbps),
        float(chol),
        fbs_mapping[fbs],
        restecg_mapping[restecg],
        float(thalach),
        exang_mapping[exang],
        float(oldpeak),
        slope_mapping[slope],
        ca_mapping[ca],
        thal_mapping[thal]
    ]

    input_data_array = np.asarray(input_data_array, dtype=np.float64)
    input_data_reshaped = input_data_array.reshape(1, -1)
    result = loaded_model.predict(input_data_reshaped)

    if result[0] == 1:
        return "The person has Cardiovascular Disease"
    else:
        return "The person does not have Cardiovascular Disease"

def main():
    st.markdown("<h1 style='text-align: center; color: red;'>Cardiovascular Disease Prediction Application</h1>", unsafe_allow_html=True)
    
    st.subheader("Patient Information:")
    
    age = st.text_input("Age of the person:")
    sex = st.selectbox("Sex:", ["Male-0", "Female-1"])
    cp = st.selectbox("Chest pain type:", ["Angina Pectoris-0", "Myocardial Infarction (Heart Attack)-1", "Gastroesophageal Reflux Disease-2", "Costochondritis-3"])
    restbps = st.text_input("Resting BP:")
    chol = st.text_input("Serum Cholestoral (mg/dl):")
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl:", ["Yes-1", "No-0"])
    restecg = st.selectbox("Resting electrocardiographic results (0-2):", ["ECG Results Regular-0", "ECG Results Severity-1", "ECG Results Indeterminate-2"])   
    thalach = st.text_input("Maximum heart rate achieved:")
    exang = st.selectbox("Exercise induced angina:", ["Yes-1", "No-0"])
    oldpeak = st.text_input("Oldpeak:")
    slope = st.selectbox("Slope of the peak exercise ST segment:", ["Upsloping ST Segment-0", "Horizontal ST Segment-1", "Downsloping ST Segment-2"])
    ca = st.selectbox("Number of major vessels (0-3):", ["No Major Coronary Vessel-0", "One Major Coronary Vessel-1", "Two Major Coronary Vessel-2", "Three  Major Coronary Vessel-3"])
    thal = st.selectbox("Thal:", ["Normal-0", "Musculoskeletal Pain-1", "Pleuritic Pain-2", "Panic Attack-3"])
    
    predict = '' # Null string
    
    if st.button('Diagnosis Test Result'):
        predict = Cardiovascular_disease_Prediction([age, sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
    st.success(predict)
    
    st.markdown("***")
    
    st.markdown("""
    Sample data to fill: 
    52 1 2 172 199 1 1 162 0 0.5 2 0 3 => Person has Cardiovascular Disease
    """)
    
    st.markdown("""
    About the data to be filled (all data is in numeric form without units):
    1. Age (in numbers)
    2. Sex (0: Female, 1: Male)
    3. Chest pain type (4 values: 0-3)
    4. Resting blood pressure (numeric only)
    5. Serum Cholestoral in mg/dl
    6. Fasting blood sugar > 120 mg/dl
    7. Resting electrocardiographic results (values 0, 1, 2)
    8. Maximum heart rate achieved
    9. Exercise induced angina
    10. Oldpeak = ST depression induced by exercise relative to rest
    11. The slope of the peak exercise ST segment
    12. Number of major vessels (0-3) colored by flourosopy
    13. Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect
    
    Output: Cardiovascular Disease Prediction (0 or 1)
    """)
    
    st.text("\n\n")

if __name__ == '__main__':
    main()
