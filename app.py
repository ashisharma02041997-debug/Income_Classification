import streamlit as st
import joblib
import numpy as np
import pandas as pd


income_model = joblib.load("xgboost_income_model.pkl")


df = pd.read_csv("income_evaluation.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove spaces from categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Replace ? with missing values
df.replace("?", np.nan, inplace=True)

# Fill missing categorical values using mode
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].fillna(df[col].mode()[0])


st.title("Income Classification Prediction Application")

st.write(
    "Enter the details below to predict whether "
    "the person's income is <=50K or >50K."
)


age = st.slider(
    "Age",
    min_value=17,
    max_value=90,
    value=30
)


workclass = st.selectbox(
    "Workclass",
    sorted(df["workclass"].unique())
)


fnlwgt = st.number_input(
    "Final Weight (fnlwgt)",
    min_value=10000,
    max_value=1500000,
    value=150000
)


education_num = st.slider(
    "Education Number",
    min_value=1,
    max_value=16,
    value=10
)


marital_status = st.selectbox(
    "Marital Status",
    sorted(df["marital-status"].unique())
)


occupation = st.selectbox(
    "Occupation",
    sorted(df["occupation"].unique())
)


relationship = st.selectbox(
    "Relationship",
    sorted(df["relationship"].unique())
)


race = st.selectbox(
    "Race",
    sorted(df["race"].unique())
)


sex = st.selectbox(
    "Sex",
    sorted(df["sex"].unique())
)


capital_gain = st.number_input(
    "Capital Gain",
    min_value=0,
    max_value=100000,
    value=0
)


capital_loss = st.number_input(
    "Capital Loss",
    min_value=0,
    max_value=5000,
    value=0
)


hours_per_week = st.slider(
    "Hours per Week",
    min_value=1,
    max_value=99,
    value=40
)


native_country = st.selectbox(
    "Native Country",
    sorted(df["native-country"].unique())
)



if st.button("Predict"):

    # Get the exact features used while training XGBoost
    model_columns = income_model.get_booster().feature_names

    # Create dataframe with all model columns initially set to 0
    final_input = pd.DataFrame(
        0,
        index=[0],
        columns=model_columns,
        dtype=float
    )



    # Apply same log transformations used during training
    numerical_values = {

        "age": age,

        "fnlwgt": np.log1p(fnlwgt),

        "education-num": education_num,

        "capital-gain": np.log1p(capital_gain),

        "capital-loss": np.log1p(capital_loss),

        "hours-per-week": hours_per_week
    }


    # Put numerical values into correct columns
    for column, value in numerical_values.items():

        if column in final_input.columns:

            final_input.loc[0, column] = value


   

    categorical_values = {

        "workclass": workclass,

        "marital-status": marital_status,

        "occupation": occupation,

        "relationship": relationship,

        "race": race,

        "sex": sex,

        "native-country": native_country
    }


    # Convert selected categories into one-hot encoded values
    for column, value in categorical_values.items():

        dummy_column = f"{column}_{value}"

        if dummy_column in final_input.columns:

            final_input.loc[0, dummy_column] = 1


 

    prediction = income_model.predict(final_input)

   


    if prediction[0] == 1:

        st.success("Predicted Income: >50K")

    else:

        st.success("Predicted Income: <=50K")


    