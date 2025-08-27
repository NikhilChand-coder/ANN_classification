import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model


st.title("Customer Churn Prediction")

#Load Model
model = load_model('model.h5')

# Loading Standardscaler,Label_encoder,One_hot_encoder
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


credit_score = st.number_input('CreditScore')
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,75,25)
tenure = st.slider("Tenure",1,12)
balance = st.number_input("Balance")
numberofproducts = st.slider("Number of Products",1,4)
hascredit = st.selectbox("How many credit card do you have",[0,1])
active_member = st.selectbox("IS active member",[0,1])
salary = st.number_input("Estimated Salary")


input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numberofproducts],
    'HasCrCard': [hascredit],
    'IsActiveMember': [active_member],
    'EstimatedSalary': [salary]
}



# Creating dataframe of input data
input_data_df = pd.DataFrame(input_data)

#Label encoder of gender
input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])

# one hot encoding and concating for geography
geo_encoded = one_hot_encoder_geo.transform(input_data_df[['Geography']]).toarray()
geo_df = pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data_df = pd.concat([input_data_df.drop(columns='Geography',axis=1),geo_df],axis=1)

# scaling
scaled_input = scaler.transform(input_data_df)

#prediction
prediction = model.predict(scaled_input)

prediction_prob = prediction[0][0]

st.write(f"Prediction probability: {prediction_prob}")

#final result
if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')


