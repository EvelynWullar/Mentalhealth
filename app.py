import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

le_industry = LabelEncoder()
le_location = LabelEncoder()
le_sleep_quality = LabelEncoder()
le_stress_level = LabelEncoder()
le_mental_health_resources = LabelEncoder()
le_physical_activity = LabelEncoder()
le_region = LabelEncoder()

# Load the trained model
model = joblib.load("my_model.pkl")
   

# Streamlit app layout
st.title('Mental Health Condition Prediction')

# Input fields for the user
age = st.number_input('Age', min_value=18, max_value=60, value=30,step=1)
hours_worked_per_week = st.number_input('Hours Worked Per Week', min_value=0, max_value=100, value=40, step=1)
mental_health_condition = st.selectbox('Mental Health Condition', options=['Depression', 'Anxiety', 'Burnout'])
gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])
industry = st.selectbox('Industry', options=['Healthcare', 'IT', 'Consulting','Manufacturing','Retail','Finance','Education'])
location = st.selectbox('Location', options=['Remote', 'Hybrid', 'Onsite'])
sleep_quality = st.selectbox('Sleep Quality', options=['Good','Poor','Average'])
stress_level = st.selectbox('Stress Level', options=['High','Medium','Low'])
mental_health_resources = st.selectbox('Access to Mental Health Resources', options=['Yes','No'])
physical_activity = st.selectbox('Physical Activity', options=['Daily','Weekly'])
region = st.selectbox('Region', options=['Africa', 'Asia', 'North America','South America','Oceania','Europe'])

industry_encoded = le_industry.fit(['Healthcare', 'IT', 'Consulting','Manufacturing','Retail','Finance','Education']).transform([industry])[0]
location_encoded = le_location.fit(['Remote', 'Hybrid', 'Onsite']).transform([location])[0]
sleep_quality_encoded = le_sleep_quality.fit(['Good','Poor','Average']).transform([sleep_quality])[0]
stress_level_encoded = le_stress_level.fit(['High','Medium','Low']).transform([stress_level])[0]
mental_health_resources_encoded = le_mental_health_resources.fit(['Yes','No']).transform([mental_health_resources])[0]
physical_activity_encoded = le_physical_activity.fit(['Daily','Weekly']).transform([physical_activity])[0]
region_encoded = le_region.fit(['Africa', 'Asia', 'North America','South America','Oceania','Europe']).transform([region])[0]

# Display the encoded values
st.write(f"Encoded Industry: {industry_encoded}")
st.write(f"Encoded Location: {location_encoded}")
st.write(f"Encoded Sleep Quality: {sleep_quality_encoded}")
st.write(f"Encoded Stress Level: {stress_level_encoded}")
st.write(f"Encoded Mental Health Resources: {mental_health_resources_encoded}")
st.write(f"Encoded Physical Activity: {physical_activity_encoded}")
st.write(f"Encoded Region: {region_encoded}")

# Creating the input feature vector for prediction
user_input_data = pd.DataFrame({
    'Age': [age],
    'Hours_Worked_Per_Week': [hours_worked_per_week],
    'Mental_Health_Condition': [mental_health_condition],  # You might need to encode this too if necessary
    'Gender': [gender],  # Encode if required
    'Industry': [industry_encoded],
    'Location': [location_encoded],
    'Sleep_Quality': [sleep_quality_encoded],
    'Stress_Level': [stress_level_encoded],
    'Access_to_Mental_Health_Resources': [mental_health_resources_encoded],
    'Physical_Activity': [physical_activity_encoded],
    'Region': [region_encoded]
})

# Predict using the trained model
if st.button('Predict'):
    prediction = model.predict(user_input_data)
    st.write(f'Predicted Mental Health Condition: {prediction[0]}')
