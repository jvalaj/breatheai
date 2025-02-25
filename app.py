import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import time
import plotly.graph_objects as go
import os
from pydub import AudioSegment
from st_audiorec import st_audiorec 
def extract_features(file_path):
    try:
        if file_path.endswith(('.obb', '.webm', '.mp3', '.flac')):
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
    st.title("breatheAI")

    # Upload or Record
    st.write("Choose an option:")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac", "ogg", "webm"])
    
    with col2:
        st.write("OR Record Audio")
        recorded_audio = st_audiorec()
 
        if recorded_audio is not None:
            st.audio(recorded_audio, format='audio/wav')

    temp_file_path = None

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        st.audio(uploaded_file, format='audio/*')

    elif recorded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(recorded_audio)
            temp_file_path = tmp_file.name
        st.audio(temp_file_path, format="audio/wav")

    if temp_file_path:
        detect_button = st.button("Detect")
        if detect_button:
            st.markdown(
                """
                <style>
                .stButton>button { color: #808080; }
                </style>
                """, unsafe_allow_html=True)

            processing_placeholder = st.empty()
            processing_placeholder.write("Processing...")

            time.sleep(2)  # Simulating processing time

            prediction, y, sr = predict_from_audio(temp_file_path)
            processing_placeholder.empty()

            if prediction is not None:
                st.write(f"**Probability of Cough:** {prediction:.2f}%")
                advice = get_advice(prediction)
                st.write(f"**Advice:** {advice}")

                # Show Visualizations
                st.subheader("Audio Visualizations")
                st.plotly_chart(visualize_waveform(y, sr))
                st.plotly_chart(visualize_spectrogram(y, sr))

            os.unlink(temp_file_path)

    st.write("---")
    st.subheader("About")
    st.write("This project detects cough sounds from audio using machine learning. The model is trained on a dataset with 31 audio features from 3029 samples sourced from Kaggle. It uses a Gradient Boosting classifier to predict the likelihood of a cough in an audio clip.")

if __name__ == "__main__":
    main()
