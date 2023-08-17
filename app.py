

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained models
LR = pickle.load(open('C:/Users/arvind gowda/project/linear_regression_model.pkl', 'rb'))
DT = pickle.load(open('C:/Users/arvind gowda/project/decision_tree_model.pkl', 'rb'))
KNN = pickle.load(open('C:/Users/arvind gowda/project/knn_model.pkl', 'rb'))
LassoR = pickle.load(open('C:/Users/arvind gowda/project/lasso_regression_model.pkl', 'rb'))
RR = pickle.load(open('C:/Users/arvind gowda/project/ridge_regression_model.pkl', 'rb'))
EN = pickle.load(open('C:/Users/arvind gowda/project/elastic_net_model.pkl', 'rb'))

# Load the feature scaler (if any)
sc_X = StandardScaler()

# Fit the scaler with the training data
training_data = pd.read_csv('C:/Users/arvind gowda/project/temperature_data.csv')
feature_columns = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d', 'i_q', 'profile_id']
sc_X.fit(training_data[feature_columns])

st.header('Electric Motor Temperature')

# Create a file uploader to allow users to upload their dataset (CSV file)
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

# Create input fields for user input
ambient = st.sidebar.number_input('Enter ambient')
coolant = st.sidebar.number_input('Enter coolant')
u_d = st.sidebar.number_input('Enter u_d')
u_q = st.sidebar.number_input('Enter u_q')
i_d = st.sidebar.number_input('Enter i_d')
i_q = st.sidebar.number_input('Enter i_q')
profile_id = st.sidebar.number_input('Enter profile_id')
motor_speed = st.sidebar.button('Predict Motor Speed')

# Define feature columns
feature_columns = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d', 'i_q', 'profile_id']

if uploaded_file is not None:
    # Read the uploaded dataset
    user_data = pd.read_csv(uploaded_file)

    # Check if the user input DataFrame has the same columns as the training data for Linear Regression
    if not set(user_data.columns) == set(feature_columns):
        st.error("The uploaded dataset should have the same columns as the training data for Linear Regression.")
    else:
        # Scale the user input (if necessary)
        user_data_scaled = sc_X.transform(user_data[feature_columns])

        # Make predictions using all the models
        lr_prediction = LR.predict(user_data_scaled)[0][0]
        dt_prediction = DT.predict(user_data[feature_columns])[0][0]
        knn_prediction = KNN.predict(user_data_scaled)[0][0]

        # For Lasso and Ridge regression, you need to fit the model first
        LassoR.fit(training_data[feature_columns], training_data['motor_speed'])
        lassor_prediction = LassoR.predict(user_data_scaled)[0]

        RR.fit(training_data[feature_columns], training_data['motor_speed'])
        rr_prediction = RR.predict(user_data_scaled)[0]

        EN.fit(training_data[feature_columns], training_data['motor_speed'])
        en_prediction = EN.predict(user_data_scaled)[0]

        # Display the predictions
        st.subheader("Predicted Value Of All The Models:")
        st.write(f"Linear Regression Prediction: {lr_prediction:.2f}")
        st.write(f"Decision Tree Prediction: {dt_prediction:.2f}")
        st.write(f"KNN Prediction: {knn_prediction:.2f}")
        st.write(f"Lassor Regression Prediction: {lassor_prediction:.2f}")
        st.write(f"Ridge Regression Prediction: {rr_prediction:.2f}")
        st.write(f"Elastic Net Prediction: {en_prediction:.2f}")
else:
    # Transform user input into a DataFrame
    user_input = pd.DataFrame({
        'ambient': [ambient],
        'coolant': [coolant],
        'u_d': [u_d],
        'u_q': [u_q],
        'i_d': [i_d],
        'i_q': [i_q],
        'profile_id': [profile_id],
        'motor_speed': [motor_speed]
    })

    # Check if the user input DataFrame has the same columns as the training data for Linear Regression
    if not set(user_input.columns) == set(feature_columns):
        st.error("The user input should have the same columns as the training data for Linear Regression.")
    else:
        # Scale the user input (if necessary)
        user_input_scaled = sc_X.transform(user_input)

        # Fit Lasso and Ridge regression with the training data
        LassoR.fit(training_data[feature_columns], training_data['motor_speed'])
        RR.fit(training_data[feature_columns], training_data['motor_speed'])
        EN.fit(training_data[feature_columns], training_data['motor_speed'])

        # Make predictions based on the selected model
        if motor_speed:
            # Display the predictions for all the models
            st.subheader("Predicted Value Of All The Models:")
            st.write(f"Linear Regression Prediction: {LR.predict(user_input_scaled)[0][0]:.2f}")
            st.write(f"Decision Tree Prediction: {DT.predict(user_input_scaled)[0][0]:.2f}")
            st.write(f"KNN Prediction: {KNN.predict(user_input_scaled)[0][0]:.2f}")
            st.write(f"Lassor Regression Prediction: {LassoR.predict(user_input_scaled)[0]:.2f}")
            st.write(f"Ridge Regression Prediction: {RR.predict(user_input_scaled)[0]:.2f}")
            st.write(f"Elastic Net Prediction: {EN.predict(user_input_scaled)[0]:.2f}")
        


