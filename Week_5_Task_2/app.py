import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

pipeline_path = os.path.join(os.path.dirname(__file__), 'titanic_pipeline.pkl')
pipeline = joblib.load(pipeline_path)

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict whether they would have survived the Titanic disaster, using a Logistic Regression model trained on the original Titanic dataset.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 29)
    fare = st.number_input("Fare paid ($)", min_value=0.0, max_value=600.0, value=50.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")

st.divider()

if st.button("Predict", use_container_width=True):
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
        st.success(f"### ✅ Survived")
        st.metric(label="Model Confidence", value=f"{probability[0][1]*100:.1f}%")
    else:
        st.error(f"### ❌ Did not survive")
        st.metric(label="Model Confidence", value=f"{probability[0][0]*100:.1f}%")
