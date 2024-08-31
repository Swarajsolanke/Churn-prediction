import streamlit as st
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle 
#load the pickle files 
model=tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#streamlit app

st.title("CUSTOMER CHURN PREDICTION")
geography=st.selectbox('Geography',one_hot_encoder.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_product=st.slider("Number of products",1,4)
has_credict_card=st.selectbox('has credit card',[0,1])
is_active_member=st.selectbox('is active member',[0,1])

# prepare the input data

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_credict_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})


encoded_value=one_hot_encoder.transform([[geography]]).toarray()
print(encoded_value)
geo_data=pd.DataFrame(encoded_value,columns=one_hot_encoder.get_feature_names_out(['Geography']))
print(geo_data)
input_data=pd.concat([input_data.reset_index(drop=True),geo_data],axis=1)
print(input_data)
input_scaled=scaler.transform(input_data)
input_scaled
prediction=model.predict(input_scaled)
print(prediction)
prob=prediction[0][0]
st.write(f'churn prediction probability: {prob:.2f}')
print(prob)
if prob>0.5:
    st.write("the customer is likely to churn")
    
else:
    st.write("the customer is not likely to churn")
    

