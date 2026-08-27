import os
import streamlit as st
import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'loan_default_model.pkl')

model = joblib.load(model_path)

st.title("Loan Default Risk Predictor")
st.write("Enter applicant details to estimate the risk of serious delinquency within 2 years.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
revolving_util = st.number_input("Revolving Credit Utilization (0-1+)", min_value=0.0, max_value=2.0, value=0.3, step=0.01)
debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=3.0, value=0.3, step=0.01)
open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", min_value=0, value=5)
real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, value=1)
dependents = st.number_input("Number of Dependents", min_value=0, value=0)

st.subheader("Payment History")
late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, value=0)
late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, value=0)
late_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0)

# recreate the engineered feature exactly as done in training
total_past_due = late_30_59 + late_60_89 + late_90

if st.button("Predict Default Risk"):
    input_data = np.array([[revolving_util, age, late_30_59, debt_ratio, monthly_income,
                             open_credit_lines, late_90, real_estate_loans, late_60_89,
                             dependents, total_past_due]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Result")
    if prediction == 1:
        st.error(f"High Risk of Default — estimated probability: {probability:.1%}")
    else:
        st.success(f"Low Risk of Default — estimated probability: {probability:.1%}")
