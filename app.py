import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import time
import plotly.graph_objects as go
import os
import pandas as pd
#import matplotlib.pyplot as plt
from pydub import AudioSegment
import joblib
import warnings

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

# -------------------- Feature Extraction Function --------------------
def extract_audio_features(file_path):
    """Extracts audio features from an uploaded file."""
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")  # Suppress librosa warnings
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features = np.hstack((np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1)))

        # Ensure feature size is 32
        if len(features) != 32:
            features = np.pad(features, (0, 32 - len(features)), mode='constant')

        return features
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return np.zeros(32)  # Return placeholder features in case of an error

# -------------------- GBM Model Prediction --------------------
def predict_gbm(audio_file_path):
    """Predicts cough probability using the GBM model."""
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

# -------------------- Multi-Output Model Prediction --------------------
def predict_multi_output(audio_file_path, user_data):
    """Processes audio file and user input, then predicts multiple outputs."""
    try:
        model = joblib.load('multi_output_model.pkl')
        scaler = joblib.load('mscaler.pkl')

        # Extract audio features
        audio_features = extract_audio_features(audio_file_path)

        if len(audio_features) != 32:
            st.error(f"Feature extraction mismatch. Expected 32 features, got {len(audio_features)}.")
            return None

        # Combine user data with extracted audio features
        combined_features = np.concatenate([user_data, audio_features]).reshape(1, -1)

        # Scale the features
        combined_features_scaled = scaler.transform(combined_features)

        # Make predictions
        prediction = model.predict(combined_features_scaled)[0]

        return prediction
    except FileNotFoundError:
        st.error("Multi-output Model or scaler file not found.")
        return None
    except ValueError as e:
        st.erroSr(f"Feature size mismatch: {e}")
        return None

# -------------------- Main Streamlit App --------------------
def main():
    st.set_page_config(page_title="breatheAI", page_icon="logobreatheai.PNG",layout="centered")
    col1, col2, = st.columns([2, 9])
    with col2:
        st.title("breatheAI")
    with col1:
        st.image('logobreatheai.PNG', width=550)  # Adjust width as needed
    
    st.markdown("""
    ### Disclaimer:
    **Audio processing and visualization may take up to 60 seconds due to cloud hosting limitations on the free-tier plan. Please be patient.**
    
    ### Upload Audio & Enter Details
    Upload an audio file and provide additional health-related information for analysis.
    """)

    uploaded_file = st.file_uploader("Choose an audio file", type=None)

    # User Inputs
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

            # Predict using GBM model
            cough_probability, y, sr, mfccs, chroma, spectral_contrast = predict_gbm(temp_file_path)

            # Predict using Multi-Output model
            multi_output_prediction = predict_multi_output(temp_file_path, user_data)
            processing_placeholder.empty()

            # Display GBM Prediction
            if cough_probability is not None:
                st.write(f"**Cough Intensity Detected:** {cough_probability:.2f}%")
                if cough_probability < 30:
                    st.success("✅ Your cough intensity is low. This is likely normal, but keep monitoring your symptoms.")
                elif 30 <= cough_probability < 60:
                    st.info("⚠️ Moderate cough intensity detected. Consider rest, hydration, and monitoring for any worsening symptoms.")
                elif 60 <= cough_probability < 85:
                    st.warning("⚠️ High cough intensity detected. Consider consulting a doctor if symptoms persist.")
                else:
                    st.error("🚨 Severe cough detected! Seek medical attention as soon as possible.")

            # Display Multi-Output Model Prediction
            if multi_output_prediction is not None:
                status_labels = {0: "Healthy", 1: "Symptomatic", 2: "Sick"}
                cough_type_labels = {0: "Dry Cough", 1: "Wet Cough", 2: "Barking Cough"}
                severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe"}

                st.subheader("Prediction Results")
                st.info(f"- **Health Status:** {status_labels.get(multi_output_prediction[0], 'Unknown')}")
                st.info(f"- **Cough Type:** {cough_type_labels.get(multi_output_prediction[1], 'Unknown')}")
                st.info(f"- **Severity Level:** {severity_labels.get(multi_output_prediction[2], 'Unknown')}")

                if multi_output_prediction[2] == 2:
                    st.warning("⚠️ Severe cough detected. Please consider consulting a healthcare professional.")

            os.unlink(temp_file_path)
    st.markdown("""
       ### How It Works:
    - **Upload any audio file** in formats like WAV, MP3, FLAC, OGG, and more.
    - **Feature Extraction**: The system processes the audio to extract key features like MFCCs, chroma, and spectral contrast.
    - **Machine Learning Prediction**: A trained Gradient Boosting Model (GBM) analyzes the features and predicts the probability of a cough in the recording.
    - **Visual Representations**: The waveform and spectrogram visualizations help users understand the sound characteristics.

    ### About
    - Growing up in New Delhi, I experienced firsthand the impact of severe air pollution, especially during the winter months. The persistent smog and rising cases of respiratory illnesses made it clear how crucial it is to monitor respiratory health effectively. Over time, I realized that coughing is an everyday occurrence—often dismissed without knowing whether it signals a minor irritation or a more serious condition. 
     
    - Back in 2024, at Hack-a-Bull, I developed breatheAI as a hackathon project, initially as an experimental solution to analyze cough patterns. What started as a challenge-driven innovation quickly evolved into something with real-world potential. I realized that coughing is incredibly common, yet there’s no accessible way to objectively measure its intensity or track its progression over time.
     
    - Looking ahead, breatheAI has the potential to integrate with wearable health devices, monitor respiratory health trends, and incorporate environmental data to assess the broader impact of air quality on respiratory conditions. What began as a hackathon project is now evolving into a tool that could make proactive respiratory health monitoring more accessible and data-driven.
    ### Learn More About the Creator:
     [GitHub](https://github.com/jvalaj) | [LinkedIn](https://www.linkedin.com/in/jvalaj/)
    """)
if __name__ == "__main__":
    main()
