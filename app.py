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

def display_feature_analysis(mfccs, chroma, spectral_contrast):
    """Formats and displays feature analysis in a readable way."""
    st.subheader("Feature Analysis")
    
    # Normal reference ranges (placeholders, adjust if needed)
    normal_mfcc = [-300, 60, 20, 10, 5, 5, 2, -5, -5, 0, -1, -3, 2]
    normal_chroma = [0.6, 0.5, 0.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, 0.4, 0.6, 0.7]
    normal_spectral = [15, 11, 15, 14, 16, 16, 46]
    
    # Convert numpy arrays to lists for plotting
    user_mfcc = np.mean(mfccs, axis=1).tolist()
    user_chroma = np.mean(chroma, axis=1).tolist()
    user_spectral = np.mean(spectral_contrast, axis=1).tolist()
    
    # Function to create a comparison plot
    def plot_feature_comparison(user_values, normal_values, feature_labels, title):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(feature_labels))
        ax.bar(x, normal_values, width=0.4, label="Normal Range", alpha=0.7)
        ax.bar([i + 0.4 for i in x], user_values, width=0.4, label="User's Audio", alpha=0.7)
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(feature_labels, rotation=45, ha="right")
        ax.set_xlabel("Feature Type")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        st.pyplot(fig)
    
    # Display plots
    plot_feature_comparison(user_mfcc, normal_mfcc, list(range(1, 14)), "MFCC Comparison")
    plot_feature_comparison(user_chroma, normal_chroma, list(range(1, 13)), "Chroma Comparison")
    plot_feature_comparison(user_spectral, normal_spectral, list(range(1, 8)), "Spectral Contrast Comparison")

def predict_from_audio(audio_file_path):
    features, y, sr, mfccs, chroma, spectral_contrast = extract_features(audio_file_path)

    if features is not None:
        try:
            with open('gbm_model.pkl', 'rb') as f:
                model = pickle.load(f)

            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            return prediction[0] * 100, y, sr, mfccs, chroma, spectral_contrast
        except FileNotFoundError:
            st.error("Model or scaler file not found.")
            return None, None, None, None, None, None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

def main():
    st.set_page_config(page_title="breatheAI", layout="centered")
    st.title("breatheAI: AI-Powered Cough Detection")

    st.markdown("""
    ### ⚠️ Disclaimer:
    **Audio processing and visualization may take up to 60 seconds due to cloud hosting limitations on the free-tier plan. Please be patient.**
    
    ### Upload Audio
    Upload an audio file to analyze cough probability.
    """)
    
    uploaded_file = st.file_uploader("Choose an audio file", type=None)
    
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

            prediction, y, sr, mfccs, chroma, spectral_contrast = predict_from_audio(temp_file_path)
            processing_placeholder.empty()

            if prediction is not None:
                st.write(f"**Probability of Cough Detected:** {prediction:.2f}%")
                display_feature_analysis(mfccs, chroma, spectral_contrast)

            os.unlink(temp_file_path)
    
if __name__ == "__main__":
    main()
