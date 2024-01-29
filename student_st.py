# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:08:40 2024

@author: Arth
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('model_lin.pkl','rb'))


def main():
    
    
    # giving a title
    st.title('Student Performance Web App')
    
    
    # getting the input data from the user
  
    Extracurricular_Activities_Yes = st.text_input('Extracurricular Activities (\'1\' for yes \'0\' for no)')
    Hours_Studied= st.text_input('Total number of hours studied by student')
    Previous_Scores = st.text_input('Previous score of student')
    Sample_Question_Papers_Practiced = st.text_input('Total number of Mocks Attemped')
    Sleep_Hours = st.text_input('Sleeping hours')
    # code for Prediction
    Performance_Index = ''
    
    # creating a button for Prediction
    
    if st.button('Student Performance'):
        # convert input values to numerical types
        features = np.array([int(Extracurricular_Activities_Yes), int(Hours_Studied), int(Previous_Scores),
                             int(Sample_Question_Papers_Practiced), int(Sleep_Hours)])
        
        # make a prediction using the loaded model
        Performance_Index = loaded_model.predict(features.reshape(1, -1))[0]

    st.success(f'Predicted Performance Index: {Performance_Index}')

if __name__ == '__main__':
    main()
        
        
        
 
