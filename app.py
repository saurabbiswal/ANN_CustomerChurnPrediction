import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
from tensorflow import keras        

# Load the model
model = keras.models.load_model('model.h5')
# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# Load the encoder - geography
with open('geography_encoded.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)
# Load the label encoder - gender
with open('label_encoder_gender.pkl', 'rb') as f:
    gen_label_encoder = pickle.load(f)

# Streamlit app
st.title('Bank Customer Churn Prediction')
st.write('This app predicts whether a bank customer will leave the bank or not.')
st.write('Please enter the following information:')

# Input fields
geography = st.selectbox('Geography', geo_encoder.categories_[0])
credit_score = st.number_input('Credit Score', min_value=0, max_value=850, step=1)
gender = st.selectbox("Gender", gen_label_encoder.classes_)
age = st.slider('Age', 18, 100)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance', min_value=0.0, max_value=1000000.0, step=1000.0)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, step=1000.0)

# Prepare the input data
input_data_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gen_label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode the geography
geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out())

# Concatenate the encoded geography with the input data
input_data_df = pd.concat([input_data_df, geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data_df)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction = (prediction > 0.5).astype(int)

# Display the input data
st.subheader('Input Data')
st.write(input_data_df)

# Display the scaled input data
st.subheader('Scaled Input Data')
st.write(input_data_scaled)

# Display the prediction result
st.subheader('Prediction Result')
if prediction[0][0] == 1:
    st.write('The customer is likely to leave the bank.')
else:
    st.write('The customer is likely to stay with the bank.')

# Display the prediction probability
st.subheader('Prediction Probability')
st.write(f'Probability of leaving: {prediction[0][0]:.2f}')