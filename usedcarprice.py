import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('LinearRegressionModel1.pkl')

# Read the cleaned dataset
cars = pd.read_csv(r'C:/Users/Administrator/Documents/Python Scripts/streamlit1/Price Predict 4UsedCars/Datacleaned_cars.csv')

# Define the app layout
st.title("Used Car Price Prediction App")

# Create input fields for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    name = st.sidebar.text_input('Name', 'Maruti Wagon')
    location = st.sidebar.selectbox('Location', cars['Location'].unique())
    year = st.sidebar.number_input('Year', 2000, 2024, 2022)
    kilometers_driven = st.sidebar.number_input('Kilometers Driven', 0, 200000, 50000)
    fuel_type = st.sidebar.selectbox('Fuel Type', cars['Fuel_Type'].unique())
    transmission = st.sidebar.selectbox('Transmission', cars['Transmission'].unique())
    owner_type = st.sidebar.selectbox('Owner Type', cars['Owner_Type'].unique())
    mileage = st.sidebar.number_input('Mileage', 0.0, 100.0, 20.0)
    engine = st.sidebar.number_input('Engine', 0.0, 5000.0, 1000.0)
    power = st.sidebar.number_input('Power', 0.0, 500.0, 100.0)
    seats = st.sidebar.number_input('Seats', 2, 7, 5)
    type_ = st.sidebar.selectbox('Type', cars['Type'].unique())

    data = pd.DataFrame({
        'Name': [name],
        'Location': [location],
        'Year': [year],
        'Kilometers_Driven': [kilometers_driven],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Owner_Type': [owner_type],
        'Mileage': [mileage],
        'Engine': [engine],
        'Power': [power],
        'Seats': [seats],
        'Type': [type_]
    })
    return data

# Get user input
user_input = user_input_features()

# Make prediction
prediction = model.predict(user_input)

# Display results
st.subheader('Prediction')
st.write(f'The predicted price of the used car is: ₹{prediction[0]:,.2f}Lac')

# Optionally display the raw data
st.subheader('User Input Data')
st.write(user_input)



