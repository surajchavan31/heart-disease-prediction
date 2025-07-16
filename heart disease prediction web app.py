# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:03:49 2025

@author: DELL
"""

import numpy as np
import pickle 
import streamlit as st

# loading the saved model

with open('E:\Machine Learning Model\Deploying HeartDisease Model\heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
# creating a function for prediction

def heartdisease_prediction(input_data):
    

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting  for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return 'The person does not have heart disease'
    else:
      return 'The person has heart disease'
  
    
def main():
    
    # giving a title 
    st.title("Heart Disease Prediction Web App")
    
    # getting the input data from the user
    Age = st.number_input("enter Age")
    Sex = st.number_input("enter Sex")
    ChestPainType = st.number_input("enter ChestPainType")
    RestingBP = st.number_input("enter RestingBP")
    Cholesterol = st.number_input("enter Cholesterol")
    FastingBS = st.number_input("enter FastingBS")
    RestingECG = st.number_input("enter RestingECG")
    MaxHR = st.number_input("enter MaxHR")
    ExerciseAngina = st.number_input("enter ExerciseAngina")
    Oldpeak = st.number_input("enter Oldpeak")
    ST_Slope = st.number_input("enter ST_Slope")
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('HeartDisease Test Result'):
        diagnosis = heartdisease_prediction([Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope])
        
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
        


        