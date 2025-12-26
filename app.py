import streamlit as st
import pickle
import numpy as np

scaler = pickle.load(open("scaler.pkl" , "rb"))
model = pickle.load(open("logistic_model.pkl" , "rb"))


st.title("Diabetes Prediction App")

preg = st.number_input("Pregnancies" , 0)
glu = st.number_input("Glucose" , 0)
bp = st.number_input("Blood Pressure" , 0)
skin = st.number_input("Skin Thickness" , 0)
ins = st.number_input("Insulin" , 0)
bmi = st.number_input("BMI" , 0.0)
dpf = st.number_input("Diabetes Pedigree Function" , 0.0)
age = st.number_input("Age" , 0)


if st.button("Predict"):
    data = np.array([[preg , glu, bp,skin,ins,bmi,dpf,age]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    if result[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")
