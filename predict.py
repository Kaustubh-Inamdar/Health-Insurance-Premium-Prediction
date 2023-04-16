import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
model_file = open('insurance_premium_model.pkl', 'rb')
model = pickle.load(model_file)
ridge_file= open('ridge_premium_model.pkl', 'rb')
ridge = pickle.load(ridge_file)
scaler_file = open('insurance_premium_scaler.pkl', 'rb')
scaler = pickle.load(scaler_file)

# Define the user input function
def get_user_input():
    age = st.number_input("Enter age:", min_value=1, max_value=100, value=25)
    diabetes = st.selectbox("Do you have diabetes?", options=[False, True])
    blood_pressure = st.selectbox("Do you have high blood pressure?", options=[False, True])
    any_transplant = st.selectbox("Have you undergone any transplant?", options=[False, True])
    any_chronic_diseases = st.selectbox("Do you have any chronic diseases?", options=[False, True])
    known_allergies = st.selectbox("Do you have any known allergies?", options=[False, True])
    cancer_in_family = st.selectbox("Is there a history of cancer in your family?", options=[False, True])
    num_major_surgeries = st.number_input("Enter number of major surgeries:", min_value=0, max_value=50, value=0)
    bmi_status_normal = st.selectbox("Is your BMI status normal?", options=[False, True])
    bmi_status_obesity = st.selectbox("Is your BMI status indicating obesity?", options=[False, True])
    bmi_status_overweight = st.selectbox("Is your BMI status indicating overweight?", options=[False, True])
    bmi_status_underweight = st.selectbox("Is your BMI status indicating underweight?", options=[False, True])

    # Convert the user input into an array
    user_input = [age, diabetes, blood_pressure, any_transplant, any_chronic_diseases, known_allergies, cancer_in_family, num_major_surgeries, bmi_status_normal, bmi_status_obesity, bmi_status_overweight, bmi_status_underweight]
    input_data = np.asarray(user_input)
    input_data_reshaped = input_data.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_reshaped)

    return input_data_scaled

def predict_premium(input_data):
    predict = model.predict(input_data)
    prediction = ridge.predict(input_data)
    return predict[0], prediction[0]

# Define the Streamlit app
def app():
    st.set_page_config(page_title="Insurance Premium Prediction App", page_icon=":medical_symbol:", layout="wide")
    st.title("Insurance Premium Prediction App")
    st.write("This app predicts the insurance premium based on the user's health information.")

    # Display the user input form
    user_input = get_user_input()

    # Calculate the premium prediction and display the result
    if st.button("Predict Premium"):
         linear_prediction, ridge_prediction = predict_premium(user_input)
         st.write(f"Your insurance premium is predicted to be: {linear_prediction:,.2f} INR (Linear Model) or {ridge_prediction:,.2f} INR (Ridge Model)")

if __name__ == '__main__':
    app()