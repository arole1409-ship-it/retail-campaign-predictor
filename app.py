import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Retail Campaign Response Predictor",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("Retail Campaign Response Predictor")

st.write(
    "Upload customer data to predict the probability "
    "of responding to a marketing campaign."
)

# Load the trained Random Forest model
model_bundle = joblib.load("rf_group_aware_best_model.joblib")

model = model_bundle["model"]
feature_cols = model_bundle["preprocessing"]["feature_cols"]
nps_median = model_bundle["preprocessing"]["nps_median_imputation"]
threshold = model_bundle["threshold"]

# Upload customer data
uploaded_file = st.file_uploader(
    "Upload customer data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    input_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Customer Data")
    st.write(f"Number of records: {len(input_data)}")
    st.dataframe(input_data.head())

    # Check required columns
    missing_columns = [
        col for col in feature_cols
        if col not in input_data.columns
    ]

    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")

    elif input_data.empty:
        st.error("The uploaded CSV contains no records.")

    else:
        # Select model features in the correct order
        X = input_data[feature_cols].copy()

        # Convert features to numeric values
        for col in feature_cols:
            original = X[col]
            X[col] = pd.to_numeric(original, errors="coerce")

            invalid = X[col].isna() & original.notna()

            if invalid.any():
                st.error(f"Invalid non-numeric values in {col}.")
                st.stop()

        # Impute missing NPS values
        X["nps"] = X["nps"].fillna(nps_median)

        # Check for other missing values
        if X.isna().any().any():
            st.error(
                "Missing values found in features other than NPS. "
                "Please correct the CSV."
            )
            st.stop()

        # Check for infinite values
        if not np.isfinite(X.to_numpy()).all():
            st.error("Infinite values detected. Please correct the CSV.")
            st.stop()

        st.success("Customer data validated successfully.")

        # Generate response probabilities
        probabilities = model.predict_proba(X)

        # Identify the probability column for class 1
        positive_index = list(model.classes_).index(1)

        response_probability = probabilities[:, positive_index]

        # Apply the saved classification threshold
        predictions = (
            response_probability >= threshold
        ).astype(int)

        # Create results DataFrame
        results = input_data.copy()

        results["Response Probability (%)"] = (
            response_probability * 100
        ).round(2)

        results["Predicted Response"] = predictions

        results["Predicted Class"] = results[
            "Predicted Response"
        ].map({
            1: "Responded",
            0: "Not Responded"
        })

        st.success("Predictions generated successfully!")

#Dashboard metrics

       

         # Calculate summary statistics
        total_customers = len(results)
        responders = int(predictions.sum())
        non_responders = total_customers - responders
        average_probability = response_probability.mean() * 100

        # Display prediction summary
        st.subheader("Prediction Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Customers Analysed", total_customers)
        col2.metric("Predicted Responders", responders)
        col3.metric("Predicted Non-Responders", non_responders)
        col4.metric(
            "Average Response Probability",
            f"{average_probability:.2f}%"
        )

        # Display customer predictions
        st.subheader("Customer Predictions")
        st.dataframe(results, use_container_width=True)

        # Display predicted response distribution
        st.subheader("Campaign Response Distribution")

        chart_data = pd.DataFrame({
            "Prediction": ["Responded", "Not Responded"],
            "Customers": [responders, non_responders]
        })

        st.bar_chart(chart_data.set_index("Prediction"))

        # Display response probability distribution
        st.subheader("Response Probability Distribution")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(response_probability * 100, bins=10)
        ax.set_xlabel("Response Probability (%)")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Distribution of Predicted Response Probabilities")

        st.pyplot(fig)
        plt.close(fig)

        # Download predictions
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv_data,
            file_name="campaign_predictions.csv",
            mime="text/csv"
        )

