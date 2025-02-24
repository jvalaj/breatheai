import streamlit as st
import os
import librosa
import numpy as np
import pickle
from pydub import AudioSegment
import tempfile
import time
from PIL import Image
from io import BytesIO

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

def visualize_audio(y, sr):
    if y is None or sr is None:
        return None, None

    # Waveform Image
    waveform_image = librosa.display.waveshow(y, sr=sr, res_type='kaiser_fast')
    waveform_image_pil = Image.fromarray(np.uint8(waveform_image * 255))

    # Spectrogram Image
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    spectrogram_image = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    spectrogram_image_pil = Image.fromarray(np.uint8(spectrogram_image * 255))

    return waveform_image_pil, spectrogram_image_pil

def get_advice(probability):
    if probability < 30:
        return "Low probability of a cough detected. This could be normal breathing or other sounds."
    elif 30 <= probability < 70:
        return "Moderate probability of a cough detected. Further evaluation might be necessary."
    else:
        return "High probability of a cough detected. Consider consulting a healthcare professional."

def main():
    st.set_page_config(page_title="breatheAI", layout="centered")

    st.markdown(
        """
        <style>
        body {
            color: #f0f0f0;
            background-color: #111111;
        }
        .stButton>button {
            color: #f0f0f0;
            background-color: #333333;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stFileUploader>div>div>div {
            background-color: #333333;
            color: #f0f0f0;
            border: 1px solid #555555;
            border-radius: 5px;
        }
        .stAudio>audio {
            background-color: #333333;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("breatheAI")

    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac", "ogg", "webm"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        st.audio(uploaded_file, format='audio/*')

        features, y, sr = extract_features(temp_file_path)

        waveform_pil, spectrogram_pil = visualize_audio(y, sr)

        if waveform_pil and spectrogram_pil:
            st.image(waveform_pil, caption='Waveform')
            st.image(spectrogram_pil, caption='Spectrogram')

        detect_button = st.button("Detect")
        if detect_button:
            st.markdown(
                """
                <style>
                .stButton>button {
                    color: #808080;
                }
                </style>
                """, unsafe_allow_html=True)

            processing_placeholder = st.empty()
            processing_placeholder.write("Processing...")
            for _ in range(3):
                time.sleep(0.5)
                processing_placeholder.write("Processing. .")
                time.sleep(0.5)
                processing_placeholder.write("Processing. . .")
                time.sleep(0.5)

            prediction, _, _ = predict_from_audio(temp_file_path)
            processing_placeholder.empty()

            if prediction is not None:
                st.write(f"Probability: {prediction:.2f}%")
                advice = get_advice(prediction)
                st.write(f"**Advice:** {advice}")

            os.unlink(temp_file_path)

    st.write("---")
    st.subheader("About")
    st.write("This project aims to detect cough sounds from an audio file using machine learning. The model is trained on a dataset containing metadata and 31 audio features extracted from 3029 rows of data sourced from Kaggle. The model employs a Gradient Boosting classifier to predict the likelihood of a cough being detected in the given audio input.")
    st.write("The system processes audio files (in various formats such as `.m4a`, `.obb`, `.webm`, `.mp3`, `.flac`) and extracts relevant audio features for prediction. Extracts 31 features, including Mel-frequency cepstral coefficients (MFCCs), chroma, and spectral contrast, using the `librosa` library. Uses a Gradient Boosting classifier (`gbm_model.pkl`), trained on a Kaggle dataset with metadata and audio features. After feature extraction, the model predicts the likelihood of a cough being present in the given audio file. The system works on individual audio files and can be used for batch processing by calling the function multiple times.")
    st.write("The model was trained on a dataset from Kaggle, consisting of 3029 rows of metadata and 31 audio features. These features were extracted from raw audio files representing a variety of sounds, including coughs. The features used are Mel-frequency cepstral coefficients (MFCCs), Chroma spectral features, and Spectral contrast.")

if __name__ == "__main__":
    main()