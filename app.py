
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to leave the company."
)

st.divider()

# Customer details form
st.subheader("Enter Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")

        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )

        st.subheader("Service Information")

        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        st.subheader("Subscription Information")

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        st.subheader("Billing and Tenure")

        tenure = st.number_input(
            "Tenure (Months)", min_value=0,
            max_value=100, value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges", min_value=0.0,
            value=840.0
        )

    submit = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )


# Prediction
if submit:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer_data])

    # Convert categorical values into dummy features
    categorical_columns = [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    encoded_df = input_df.drop(
        columns=categorical_columns
    ).copy()

    for column in categorical_columns:
        for feature in feature_columns:
            prefix = column + "_"

            if feature.startswith(prefix):
                category = feature[len(prefix):]

                encoded_df[feature] = int(
                    input_df[column].iloc[0] == category
                )

    # Ensure exactly the same feature order as training
    encoded_df = encoded_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale numerical features
    numerical_cols = [
        "tenure", "MonthlyCharges", "TotalCharges"
    ]

    encoded_df[numerical_cols] = scaler.transform(
        encoded_df[numerical_cols]
    )

    # Predict
    prediction = model.predict(encoded_df)[0]

    probability = model.predict_proba(
        encoded_df
    )[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.write(
            "This customer has been classified as a potential churner. "
            "Consider customer retention strategies."
        )
    else:
        st.write(
            "This customer has been classified as likely to remain."
        )
