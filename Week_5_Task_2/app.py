import streamlit as st
import joblib
import pandas as pd
pipeline = joblib.load('titanic_pipeline.pkl')

st.title("Titanic Survival Predictor")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 29)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare paid", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
if st.button("Predict"):
    family_size = sibsp + parch + 1
    alone = 1 if family_size == 1 else 0

    input_row = pd.DataFrame([{
        'Age': age,
        'Fare': fare,
        'SibSp': sibsp,
        'Parch': parch,
        'FamilySize': family_size,
        'Pclass': pclass,
        'Sex': sex,
        'Embarked': embarked,
        'Alone': alone
    }])

    prediction = pipeline.predict(input_row)
    probability = pipeline.predict_proba(input_row)

    if prediction[0] == 1:
        st.success(f"Survived — {probability[0][1]*100:.1f}% confidence")
    else:
        st.error(f"Did not survive — {probability[0][0]*100:.1f}% confidence")