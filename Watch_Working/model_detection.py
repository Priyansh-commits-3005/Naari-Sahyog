import joblib
import streamlit as st
import pandas as pd
import serial


model = joblib.load(r'C:\codes\Projects\Naari-Sahyog\model\CNN_Model_Stress_v1.h5')

st.number_input("enter the temperature value")