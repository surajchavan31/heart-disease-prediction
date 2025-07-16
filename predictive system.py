# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 18:51:47 2025

@author: DELL
"""

import numpy as np
import pickle

# loading the saved model

with open('E:/Machine Learning Model/Deploying HeartDisease Model/heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
input_data = (40,1,1,140,289,0,1,172,0,0.0,2)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting  for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('The person does not have heart disease')
else:
  print('The person has heart disease')