# Importing required libraries
import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
import joblib
# Reading the Dataset
#cars = pd.read_csv(r"C:\Users\Administrator\Documents\Python Scripts\streamlit1\Price Predict 4UsedCars\Datacleaned_cars.csv")

# Load the dataset using a relative path
cars = pd.read_csv("Datacleaned_cars.csv")

# Fetching all the unique companies, car model names, years, and fuel types
company = cars['Type'].unique()
name = cars['Name'].unique()
year = sorted(cars['Year'].unique(), reverse=True)
fuel_type = cars['Fuel_Type'].unique()

st.title("USED CAR PRICE PREDICTOR")
st.subheader("Enter the details of car: ")

# Taking user inputs
col1, col2 = st.columns(2)
with col1:
    selected_company = st.selectbox('Select Company:', company)
    
with col2:
    selected_company_models = []
    for i in list(name):
        if i.startswith(selected_company):
            selected_company_models.append(i)
            
    selected_model = st.selectbox('Select Model Name:', selected_company_models)
    
col3, col4 = st.columns(2)
with col3:
    selected_year = st.selectbox('Select Year:', year)
    
with col4:
    selected_fuel_type = st.selectbox('Select Fuel Type:', fuel_type)

selected_km_driven = st.number_input("Enter Kilometers Driven: ")

#no need below

selected_transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
selected_owner_type = st.sidebar.selectbox('Owner Type', ['First', 'Second', 'Third', 'Fourth & Above'])
selected_mileage = st.sidebar.number_input('Mileage (in km/l)', 0.0, 100.0, 0.0)
selected_engine = st.sidebar.number_input('Engine (in cc)', 0.0, 5000.0, 0.0)
selected_power = st.sidebar.number_input('Power (in bhp)', 0.0, 500.0, 0.0)
selected_seats = st.sidebar.number_input('Seats', 2, 7, 5)
selected_type = st.sidebar.selectbox('Type', ['Maruti', 'Hyundai', 'Honda','Toyota', 'Audi', 'Mahindra', 'Chevrolet'])

btn = st.button('Submit Info')
#model = joblib.load('LinearRegressionModel2.pkl')
model = joblib.load('LinearRegressionModel2.pkl')
if btn:
    # Loading trained machine learning model via pickle
    with open('LinearRegressionModel2.pkl', 'rb') as file:
        model = pkl.load(file)
    
    # Predicting the result
    prediction = model.predict(pd.DataFrame([[selected_model, selected_company, selected_year, selected_km_driven, selected_fuel_type, 
                                           selected_transmission, selected_owner_type, selected_mileage, 
                                           selected_engine, selected_power, selected_seats, selected_type]], columns=['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Type']))
    
    # Showing Results
    st.subheader(round(prediction[0], 2))

# Input data matching the expected column structure

input_data = np.array([
    'Maruti Wagon',  # Name
    'Mumbai',        # Location
    2010,            # Year
    72000,           # Kilometers_Driven
    'CNG',           # Fuel Type
    'Manual',        # Transmission
    'First',         # Owner Type
    26.6,            # Mileage
    998,             # Engine
    58.16,           # Power
    5,               # Seats
    'Maruti'         # Brand Type
]).reshape(1, 12)

# Create DataFrame ensuring column names match
expected_columns = ['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Type']
input_df = pd.DataFrame(columns=expected_columns, data=input_data)

# Ensure that numeric columns are of the correct type
input_df['Year'] = input_df['Year'].astype(int)
input_df['Mileage'] = input_df['Mileage'].astype(float)
input_df['Engine'] = input_df['Engine'].astype(float)
input_df['Kilometers_Driven'] = input_df['Kilometers_Driven'].astype(int)
input_df['Power'] = input_df['Power'].astype(float)
input_df['Seats'] = input_df['Seats'].astype(int)

# Now predict


predicted_price = model.predict(input_df)
print(predicted_price)
