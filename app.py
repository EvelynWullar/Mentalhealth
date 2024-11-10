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
    numeric_columns = ['Age', 'Hours_Worked_Per_Week']
    
    inputs[numeric_columns] = scaler.fit_transform(inputs[numeric_columns]) 

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
age = st.number_input('Age', min_value=18, max_value=60, value=30)
hours_worked = st.number_input('Hours Worked Per Week', min_value=20, max_value=60, value=40)
gender = st.selectbox('Gender', options=['Male', 'Female'])
industry = st.selectbox('Industry', options=['Healthcare', 'IT', 'Consulting','Manufacturing','Retail','Finance','Education']) 
location = st.selectbox('Location', options=['Remote', 'Hybrid', 'Onsite']) 
sleep_quality = st.selectbox('Sleep Quality', options=['Good','Poor','Average'])
stress_level = st.selectbox('Stress Level', options=['High','Medium','Low'])
mental_health_resources = st.selectbox('Access to Mental Health Resources', options=['Yes','No'])
physical_activity = st.selectbox('Physical Activity', options=['Daily','Weekly'])
region = st.selectbox('Region', options=['Africa', 'Asia', 'North America','South America','Oceania','Europe']) 

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

