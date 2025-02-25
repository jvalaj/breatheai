import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import time
import plotly.graph_objects as go
import os
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
        return features, y, sr
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return None, None, None

def predict_from_audio(audio_file_path):
    features, y, sr = extract_features(audio_file_path)

    if features is not None:
        try:
            with open('gbm_model.pkl', 'rb') as f:
                model = pickle.load(f)

            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            return prediction[0] * 100, y, sr
        except FileNotFoundError:
            st.error("Model or scaler file not found.")
            return None, None, None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None, None, None
    else:
        return None, None, None

def visualize_waveform(y, sr):
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', name='Waveform'))
    fig.update_layout(title="Waveform", xaxis_title="Time (seconds)", yaxis_title="Amplitude", template="plotly_dark")
    return fig

def visualize_spectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=D, colorscale='Inferno'))
    fig.update_layout(title="Spectrogram", xaxis_title="Time", yaxis_title="Frequency", template="plotly_dark")
    return fig

def get_advice(probability):
    if probability < 30:
        return "Low probability of a cough detected. This could be normal breathing or other sounds."
    elif 30 <= probability < 80:
        return "Moderate probability of a cough detected. Further evaluation might be necessary."
    else:
        return "High probability of a cough detected. Consider consulting a healthcare professional."

def main():
    st.set_page_config(page_title="breatheAI", layout="centered")
    st.title("breatheAI: AI-Powered Cough Detection")

    st.markdown("""
    ### ⚠️ Disclaimer:
    **Audio processing and visualization may take up to 60 seconds due to cloud hosting limitations on the free-tier plan. Please be patient.**
    
    ### Upload Audio
    Upload an audio file to analyze cough probability.
    """)
    
    uploaded_file = st.file_uploader("Choose an audio file", type=None)  # Accepts any audio format
    
    temp_file_path = None

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        st.audio(uploaded_file, format='audio/*')
    
    if temp_file_path:
        detect_button = st.button("Analyze Audio")
        if detect_button:
            st.markdown(
                """
                <style>
                .stButton>button { color: #808080; }
                </style>
                """, unsafe_allow_html=True)

            processing_placeholder = st.empty()
            processing_placeholder.write("Analyzing audio file...")

            time.sleep(2)  # Simulating processing time

            prediction, y, sr = predict_from_audio(temp_file_path)
            processing_placeholder.empty()

            if prediction is not None:
                st.write(f"**Probability of Cough Detected:** {prediction:.2f}%")
                advice = get_advice(prediction)
                st.write(f"**Medical Advice:** {advice}")

                # Show Visualizations
                st.subheader("Audio Analysis & Visualization")
                st.plotly_chart(visualize_waveform(y, sr))
                st.plotly_chart(visualize_spectrogram(y, sr))

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
    - [GitHub](https://github.com/jvalaj)
    - [LinkedIn](https://www.linkedin.com/in/jvalaj/)
    """)

if __name__ == "__main__":
    main()
