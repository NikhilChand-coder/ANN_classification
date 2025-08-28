import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model


st.title("Churn Customer Estimated Salary Prediction")

#Load Model
model = load_model('model_reg.h5')

# Loading Standardscaler,Label_encoder,One_hot_encoder
with open('reg_label_encoder_gender.pkl','rb') as file:
    reg_label_encoder_gender = pickle.load(file)

with open('reg_one_hot_encoder_geo.pkl','rb') as file:
    reg_one_hot_encoder_geo = pickle.load(file)

with open('reg_scaler.pkl','rb') as file:
    reg_scaler = pickle.load(file)


credit_score = st.number_input('CreditScore')
geography = st.selectbox("Geography", reg_one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",reg_label_encoder_gender.classes_)
age = st.slider("Age",18,75,25)
tenure = st.slider("Tenure",1,12)
balance = st.number_input("Balance")
numberofproducts = st.slider("Number of Products",1,4)
hascredit = st.selectbox("How many credit card do you have",[0,1])
active_member = st.selectbox("IS active member",[0,1])
exited = st.selectbox("Exited",[0,1])


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
    'Exited': [exited]
}



# Creating dataframe of input data
input_data_df = pd.DataFrame(input_data)

#Label encoder of gender
input_data_df['Gender'] = reg_label_encoder_gender.transform(input_data_df['Gender'])

# one hot encoding and concating for geography
geo_encoded = reg_one_hot_encoder_geo.transform(input_data_df[['Geography']]).toarray()
geo_df = pd.DataFrame(geo_encoded,columns=reg_one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data_df = pd.concat([input_data_df.drop(columns='Geography',axis=1),geo_df],axis=1)

# scaling
scaled_input = reg_scaler.transform(input_data_df)

#prediction
prediction = model.predict(scaled_input)

#final result
st.write(f"Estimated Salary: {prediction}")



