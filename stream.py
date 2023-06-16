# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:39:55 2023

@author: debna
"""

import streamlit as st
import numpy as np
import pickle
from pickle import load
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import numpy as np
import pickle
from pickle import load
from sklearn.preprocessing import MinMaxScaler

# Load the model and scaler
lr = load(open('Virtual_Reality_dt_new.pickle', 'rb'))
scaler = load(open('MinMaxScaler_new.pickle', 'rb'))

st.title("[Virtual Reality Performances]")

Age = st.number_input("[Enter Your age]", 0, 100)

Gender = st.radio("[Select Your Gender]", ["Male", "Female", "Other"])

if Gender == "Male":
    Gender = 1
elif Gender == "Female":
    Gender = 2
elif Gender == "Other":
    Gender = 3

VRHeadset = st.radio("[Select VRHeadset]", ["Oculus Rift", "HTC Vive", "PlayStation VR"])

if VRHeadset == "Oculus Rift":
    VRHeadset = 1
elif VRHeadset == "HTC Vive":
    VRHeadset = 2
elif VRHeadset == "PlayStation VR":
    VRHeadset = 3

MotionSickness = st.slider("[Enter MotionSickness]", 0, 100)
Duration = st.number_input("[Enter Duration]", 0, 100)

if st.button("Predict"):
    query_point = np.array([Age, Gender, VRHeadset, MotionSickness, Duration])
    query_point = query_point.reshape(1, -1)
    query_point_transformed = scaler.transform(query_point)
    prediction = lr.predict(query_point_transformed)
    st.write("The Performance score is", prediction[0])



##streamlit run stream.py & npx localtunnel --port 8501 


 








