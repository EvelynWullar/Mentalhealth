import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("my_model.pkl")

# Function to preprocess the inputs
def preprocess_inputs(inputs):
    # Example of scaling numerical inputs (you need to know which ones to scale)
    scaler = StandardScaler()
    
    # Assuming `inputs` is a pandas DataFrame with correct input column names
    # Scale numerical columns
    numeric_columns = ['Age', 'Hours_Worked_Per_Week', 'sleep_Quality', 'stress_Level', 
                       'access_to_Mental_Health_Resource', 'physical_Activity']
    
    inputs[numeric_columns] = scaler.fit_transform(inputs[numeric_columns])  # Scale the numerical columns

    # Handle categorical data if needed, e.g., using one-hot encoding or manual encoding for 'gender', 'industry', etc.
    gender_map = {'Male': 0, 'Female': 1}
    inputs['gender'] = inputs['gender'].map(gender_map)
    
    return inputs

# Function to predict mental health condition
def predict_mental_health_condition(user_input):
    # Preprocess the input before making a prediction
    processed_input = preprocess_inputs(user_input)
    
    # Extract features from the processed input (exclude target variable if present)
    features = processed_input.drop(columns='Mental_Health_Condition')
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Map the prediction to the mental health condition
    conditions = ['Depression', 'Anxiety', 'Burnout']
    return conditions[prediction[0]]

# Streamlit app layout
st.title('Mental Health Condition Prediction')

# Input fields for the user
age = st.number_input('Age', min_value=18, max_value=100, value=30)
hours_worked = st.number_input('Hours Worked Per Week', min_value=0, max_value=168, value=40)
gender = st.selectbox('Gender', options=['Male', 'Female'])
industry = st.selectbox('Industry', options=['Industry1', 'Industry2', 'Industry3'])  # Add actual industries
location = st.selectbox('Location', options=['Location1', 'Location2', 'Location3'])  # Add actual locations
sleep_quality = st.number_input('Sleep Quality (1-10)', min_value=1, max_value=10, value=7)
stress_level = st.number_input('Stress Level (1-10)', min_value=1, max_value=10, value=5)
mental_health_resources = st.number_input('Access to Mental Health Resources (1-10)', min_value=1, max_value=10, value=6)
physical_activity = st.number_input('Physical Activity (1-10)', min_value=1, max_value=10, value=7)
region = st.selectbox('Region', options=['Region1', 'Region2', 'Region3'])  # Add actual regions

# Create a dataframe for user inputs
user_input = pd.DataFrame({
    'Age': [age],
    'Hours_Worked_Per_Week': [hours_worked],
    'Mental_Health_Condition': [0],  # This column will not be used for prediction
    'gender': [gender],
    'industry': [industry],
    'Location': [location],
    'sleep_Quality': [sleep_quality],
    'stress_Level': [stress_level],
    'access_to_Mental_Health_Resource': [mental_health_resources],
    'physical_Activity': [physical_activity],
    'region': [region]
})

# Predict button
if st.button('Predict'):
    prediction = predict_mental_health_condition(user_input)
    st.write(f'The predicted mental health condition is: {prediction}')

