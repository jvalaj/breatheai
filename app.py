import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import time
import plotly.graph_objects as go
import os
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
import joblib

def extract_features(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            file_path = tmp_file.name
            audio.export(file_path, format='wav')

        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features = np.hstack((np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1)))
        return features, y, sr, mfccs, chroma, spectral_contrast
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return None, None, None, None, None, None

def predict_gbm(audio_file_path):
    features, y, sr, mfccs, chroma, spectral_contrast = extract_features(audio_file_path)
    
    if features is not None:
        try:
            with open('gbm_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('gscaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0] * 100
            return prediction, y, sr, mfccs, chroma, spectral_contrast
        except FileNotFoundError:
            st.error("GBM Model or scaler file not found.")
            return None, None, None, None, None, None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

def predict_multi_output(audio_features, user_data):
    try:
        model = joblib.load('multi_output_model.pkl')
        scaler = joblib.load('mscaler.pkl')
        
        user_data = np.concatenate([user_data, audio_features])
        user_data_scaled = scaler.transform([user_data])
        prediction = model.predict(user_data_scaled)[0]
        return prediction
    except FileNotFoundError:
        st.error("Multi-output Model or scaler file not found.")
        return None
    except ValueError as e:
        st.error(f"Feature size mismatch: {e}")
        return None

def main():
    st.set_page_config(page_title="breatheAI", layout="centered")
    st.title("breatheAI: AI-Powered Cough Detection")

    st.markdown("""
    ### ⚠️ Disclaimer:
    **Audio processing and visualization may take up to 60 seconds due to cloud hosting limitations on the free-tier plan. Please be patient.**
    
    ### Upload Audio & Enter Details
    Upload an audio file and provide additional health-related information for analysis.
    """)
    
    uploaded_file = st.file_uploader("Choose an audio file", type=None)
    
    age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
    gender = st.radio("Select gender:", options=["Male", "Female", "Other"])
    fever_muscle_pain = st.radio("Do you have fever/muscle pain?", options=["No", "Yes"])
    respiratory_condition = st.radio("Do you have a respiratory condition?", options=["No", "Yes"])
    
    gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
    fever_mapping = {"No": 0, "Yes": 1}
    respiratory_mapping = {"No": 0, "Yes": 1}
    user_data = np.array([age, gender_mapping[gender], fever_mapping[fever_muscle_pain], respiratory_mapping[respiratory_condition]])
    
    temp_file_path = None

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        st.audio(uploaded_file, format='audio/*')
    
    if temp_file_path:
        detect_button = st.button("Analyze Audio")
        if detect_button:
            processing_placeholder = st.empty()
            processing_placeholder.write("Analyzing audio file...")
            
            time.sleep(2)
            
            cough_probability, y, sr, mfccs, chroma, spectral_contrast = predict_gbm(temp_file_path)
            audio_features = np.mean(mfccs, axis=1).tolist() if mfccs is not None else [0] * 13
            multi_output_prediction = predict_multi_output(audio_features, user_data)
            processing_placeholder.empty()
            
            if cough_probability is not None:
                st.write(f"**Probability of Cough Detected:** {cough_probability:.2f}%")
            
            if multi_output_prediction is not None:
                status_labels = {0: "Healthy", 1: "Symptomatic", 2: "Sick"}
                cough_type_labels = {0: "Dry Cough", 1: "Wet Cough", 2: "Barking Cough"}
                severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe"}
                
                st.subheader("Prediction Results")
                st.write(f"- **Health Status:** {status_labels.get(multi_output_prediction[0], 'Unknown')}")
                st.write(f"- **Cough Type:** {cough_type_labels.get(multi_output_prediction[1], 'Unknown')}")
                st.write(f"- **Severity Level:** {severity_labels.get(multi_output_prediction[2], 'Unknown')}")
                
                if multi_output_prediction[2] == 2:
                    st.warning("⚠️ Severe cough detected. Please consider consulting a healthcare professional.")
            
            os.unlink(temp_file_path)
    st.markdown("""
       ### How It Works:
    - **Upload any audio file** in formats like WAV, MP3, FLAC, OGG, and more.
    - **Feature Extraction**: The system processes the audio to extract key features like MFCCs, chroma, and spectral contrast.
    - **Machine Learning Prediction**: A trained Gradient Boosting Model (GBM) analyzes the features and predicts the probability of a cough in the recording.
    - **Visual Representations**: The waveform and spectrogram visualizations help users understand the sound characteristics.
        ### Inspiration for breatheAI
    Growing up in New Delhi, I witnessed firsthand how intense air pollution could be, especially during the winter months. 
    The thick smog, the difficulty in breathing, and the rising cases of respiratory illnesses always had me thinking—how can technology help?
    As a college student, I realized the potential of AI and machine learning in identifying respiratory conditions early. 
    That’s when I decided to build breatheAI, a tool that could detect coughs from audio recordings and provide real-time insights. 
    Hopefully, this project can contribute to a future where technology aids in monitoring air-quality-related illnesses and helps people make informed health decisions.
    
    ### Learn More About the Creator:
     [GitHub](https://github.com/jvalaj) | [LinkedIn](https://www.linkedin.com/in/jvalaj/)
    """)
if __name__ == "__main__":
    main()