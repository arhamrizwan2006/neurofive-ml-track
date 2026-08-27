import os
import streamlit as st
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'loan_default_model.pkl'))

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳", layout="centered")

st.title("💳 Loan Default Risk Predictor")
st.write("Estimate the risk that an applicant will experience serious delinquency within 2 years.")

with st.expander("ℹ️ About this model"):
    st.write(
        "This model was trained on the 'Give Me Some Credit' dataset (~150,000 records) "
        "using a Random Forest classifier. It predicts the probability of serious loan "
        "delinquency based on the applicant's financial profile and payment history."
    )

st.subheader("Personal & Financial Info")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
    dependents = st.number_input("Number of Dependents", min_value=0, value=0)
with col2:
    revolving_util = st.number_input("Revolving Credit Utilization (0-1+)", min_value=0.0, max_value=2.0, value=0.3, step=0.01)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=3.0, value=0.3, step=0.01)

col3, col4 = st.columns(2)
with col3:
    open_credit_lines = st.number_input("Open Credit Lines/Loans", min_value=0, value=5)
with col4:
    real_estate_loans = st.number_input("Real Estate Loans/Lines", min_value=0, value=1)

st.subheader("Payment History")
col5, col6, col7 = st.columns(3)
with col5:
    late_30_59 = st.number_input("30-59 Days Late", min_value=0, value=0)
with col6:
    late_60_89 = st.number_input("60-89 Days Late", min_value=0, value=0)
with col7:
    late_90 = st.number_input("90+ Days Late", min_value=0, value=0)

total_past_due = late_30_59 + late_60_89 + late_90

st.divider()

if st.button("Predict Default Risk", type="primary", use_container_width=True):
    input_data = np.array([[revolving_util, age, late_30_59, debt_ratio, monthly_income,
                             open_credit_lines, late_90, real_estate_loans, late_60_89,
                             dependents, total_past_due]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")
    st.progress(min(probability, 1.0))

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Risk of Default — estimated probability: {probability:.1%}")
