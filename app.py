import streamlit as st
import joblib
import numpy as np
from process import process_audio

# Load your trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Cough Detection App')
audio_file = st.file_uploader('Upload an audio file', type=['wav', 'mp3'])

if audio_file is not None:
    # Process the audio file and make predictions
    features = process_audio(audio_file, scaler)
    prediction = model.predict(features)
    st.write(f'Prediction: {prediction[0]}')