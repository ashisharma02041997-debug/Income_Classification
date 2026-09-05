import streamlit as st
import joblib
import pandas as pd
import numpy as np


# Load trained XGBoost model and scaler
model = joblib.load("xgboost_income_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = joblib.load("feature_names.pkl")

# Title
st.title("Income Classification Application")


# User Inputs

age = st.slider(
    "Age",
    17,
    90,
    40
)

workclass = st.selectbox(
    "Workclass",
    [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked"
    ]
)

fnlwgt = st.number_input(
    "Final Weight (fnlwgt)",
    min_value=1,
    value=200000
)

education_num = st.slider(
    "Education Number",
    1,
    16,
    10
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces"
    ]
)

relationship = st.selectbox(
    "Relationship",
    [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried"
    ]
)

race = st.selectbox(
    "Race",
    [
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other"
    ]
)

sex = st.selectbox(
    "Sex",
    [
        "Male",
        "Female"
    ]
)

capital_gain = st.number_input(
    "Capital Gain",
    min_value=0,
    value=0
)

capital_loss = st.number_input(
    "Capital Loss",
    min_value=0,
    value=0
)

hours_per_week = st.slider(
    "Hours per Week",
    1,
    99,
    40
)

native_country = st.selectbox(
    "Native Country",
    [
        "United-States",
        "Mexico",
        "Philippines",
        "Germany",
        "Canada",
        "India",
        "Cuba",
        "England",
        "Jamaica",
        "China",
        "Japan",
        "Italy",
        "France",
        "Other"
    ]
)


# Prediction

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "fnlwgt": [fnlwgt],
        "education-num": [education_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "sex": [sex],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country]
    })


    # Same transformations used during model training
    input_data["fnlwgt"] = np.log1p(input_data["fnlwgt"])
    input_data["capital-gain"] = np.log1p(input_data["capital-gain"])
    input_data["capital-loss"] = np.log1p(input_data["capital-loss"])


    # Categorical columns
    categorical_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]


    # One-hot encoding
    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )


    # Match exact training features
    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )


    # Scale input
    input_scaled = scaler.transform(input_encoded)


    # XGBoost prediction
    result = model.predict(input_scaled)


    # Show result
    if result[0] == 1:
        st.success("Predicted Income: >50K")

    else:
        st.success("Predicted Income: <=50K")