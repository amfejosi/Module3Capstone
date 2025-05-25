import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and threshold
with open('c:/Users/amfed/OneDrive/ドキュメント/Purwadhika/Module 3/Module3Capstone/final_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

THRESHOLD = 0.4

st.title("Customer Outcome Prediction App")
st.write("Enter customer data to predict the outcome (churn or not churn).")

# inputs
tenure = st.number_input("Tenure", min_value=0)
warehouse_to_home = st.number_input("Warehouse To Home", min_value=5)
number_of_device_registered = st.number_input("Number of Device Registered", min_value=1)
prefered_order_cat = st.selectbox("Prefered Order Category", options=["Laptop & Accessory", "Mobile", "Fashion", "Mobile Phone", "Grocery", "Others"])
satisfaction_score = st.number_input("Satisfaction Score", min_value = 1, max_value=5)
marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
number_of_address = st.number_input("Number of Address", min_value=1)
complain = st.radio("Complain", options=[0, 1])
day_since_last_order = st.number_input("Day Since Last Order", min_value=0)
cashback_amount = st.number_input("Cashback Amount", min_value=0)

# Convert input to DataFrame (adjust columns to match training data)
input_data = pd.DataFrame([{
    "Tenure": tenure,
    "WarehouseToHome": warehouse_to_home,
    "NumberOfDeviceRegistered": number_of_device_registered,
    "PreferedOrderCat": prefered_order_cat,
    "SatisfactionScore": satisfaction_score,
    "MaritalStatus": marital_status,
    "NumberOfAddress": number_of_address,
    "Complain": complain,
    "DaySinceLastOrder": day_since_last_order,
    "CashbackAmount": cashback_amount
}])

if st.button("Predict"):
    # Predict probabilities
    prob = model.predict_proba(input_data)[0][1]

    # Apply threshold
    prediction = int(prob >= THRESHOLD)

    # Output
    st.write(f"**Probability of outcome = 1 (customer will churn):** {prob:.3f}")
    st.write(f"**Predicted class at threshold {THRESHOLD}:** {prediction}")

    if prediction == 1:
        st.success("⚠️ Prediction: Likely to churn")
    else:
        st.info("✅ Prediction: Likely to not churn")