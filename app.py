import streamlit as st
import pickle
import pandas as pd

# Load the best model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Streamlit App Title
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# User Inputs
rating = st.slider('Rating', 1.0, 5.0, 3.5)

company_name_options = label_encoders['Company Name'].classes_
company_name_selected = st.selectbox('Company Name', company_name_options)

job_title_options = label_encoders['Job Title'].classes_
job_title_selected = st.selectbox('Job Title', job_title_options)

salaries_reported = st.number_input('Salaries Reported', min_value=1, value=5)

location_options = label_encoders['Location'].classes_
location_selected = st.selectbox('Location', location_options)

employment_status_options = label_encoders['Employment Status'].classes_
employment_status_selected = st.selectbox('Employment Status', employment_status_options)

job_role_options = label_encoders['Job Roles'].classes_
job_role_selected = st.selectbox('Job Roles', job_role_options)

# Preprocess user input
try:
    company_name_encoded = label_encoders['Company Name'].transform([company_name_selected])[0]
    job_title_encoded = label_encoders['Job Title'].transform([job_title_selected])[0]
    location_encoded = label_encoders['Location'].transform([location_selected])[0]
    employment_status_encoded = label_encoders['Employment Status'].transform([employment_status_selected])[0]
    job_role_encoded = label_encoders['Job Roles'].transform([job_role_selected])[0]
except ValueError as e:
    st.error(f"Error encoding input: {e}. Please ensure all selected options were present during model training.")
    st.stop()

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'Rating': [rating],
    'Company Name': [company_name_encoded],
    'Job Title': [job_title_encoded],
    'Salaries Reported': [salaries_reported],
    'Location': [location_encoded],
    'Employment Status': [employment_status_encoded],
    'Job Roles': [job_role_encoded]
})

# Predict button
if st.button('Predict Salary'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ₹{prediction:,.2f}')
