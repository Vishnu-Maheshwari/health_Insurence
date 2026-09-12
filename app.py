import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("🏥 Health Insurance Cost Predictor")
st.write(
    "Enter your personal details below to predict your estimated "
    "health insurance charges."
)

st.divider()


# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

smoker = st.selectbox(
    "Do you smoke?",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)


# -----------------------------
# Prediction Button
# -----------------------------

if st.button("🔮 Predict Insurance Cost"):

    # Gender encoding
    if gender == "Female":
        is_female = 1
    else:
        is_female = 0

    # Smoker encoding
    if smoker == "Yes":
        is_smoker = 1
    else:
        is_smoker = 0

    # BMI Category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"


    # -----------------------------
    # Create Input DataFrame
    # -----------------------------

    input_data = pd.DataFrame({
        "age": [age],
        "is_female": [is_female],
        "bmi": [bmi],
        "children": [children],
        "is_smoker": [is_smoker],
        "region_southeast": [
            1 if region == "Southeast" else 0
        ],
        "bmi_category_Obese": [
            1 if bmi_category == "Obese" else 0
        ]
    })


    # -----------------------------
    # Scale Numerical Columns
    # -----------------------------

    numerical_columns = [
        "age",
        "bmi",
        "children"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )


    # -----------------------------
    # Make Prediction
    # -----------------------------

    # Ensure the input has exactly the same feature names and order
# as the trained model.
if hasattr(model, "feature_names_in_"):
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

prediction = model.predict(input_data)[0]


    # -----------------------------
    # Display Result
    # -----------------------------

    st.success("Prediction completed successfully! 🎉")

    st.subheader("💰 Estimated Insurance Cost")

    st.metric(
        label="Predicted Charges",
        value=f"${prediction:,.2f}"
    )

    st.info(
        "Note: This is an estimated prediction based on the trained "
        "machine learning model."
    )
