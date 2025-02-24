import streamlit as st
import os
import librosa
import numpy as np
import pickle
from pydub import AudioSegment
import tempfile
import time
from io import BytesIO
from PIL import Image

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

        if st.button("Detect"):
            processing_placeholder = st.empty()
            processing_placeholder.write("Processing...")
            for _ in range(3):
                time.sleep(0.5)
                processing_placeholder.write("Processing. .")
                time.sleep(0.5)
                processing_placeholder.write("Processing. . .")
                time.sleep(0.5)

            prediction, y, sr = predict_from_audio(temp_file_path)
            processing_placeholder.empty()

            if prediction is not None:
                st.write(f"Probability: {prediction:.2f}%")

                # Waveform Image
                waveform_image = librosa.display.waveshow(y, sr=sr, res_type='kaiser_fast')
                waveform_image_pil = Image.fromarray(np.uint8(waveform_image * 255))
                st.image(waveform_image_pil, caption='Waveform')

                # Spectrogram Image
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                spectrogram_image = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
                spectrogram_image_pil = Image.fromarray(np.uint8(spectrogram_image * 255))
                st.image(spectrogram_image_pil, caption='Spectrogram')

            os.unlink(temp_file_path)

    st.write("---")
    st.subheader("About breatheAI")
    st.write("This model analyzes audio files to detect coughs. It extracts features such as MFCCs, Chroma, and Spectral Contrast. These features are then scaled and fed into a Gradient Boosting Machine (GBM) model to predict the probability of a cough.")
    st.write("The model has been trained on a diverse dataset of cough and non-cough audio samples to ensure accuracy.")
    st.write("MFCCs (Mel-frequency cepstral coefficients) represent the short-term power spectrum of a sound, Chroma features relate to pitch content, and Spectral Contrast captures the differences in spectral peaks and valleys.")

if __name__ == "__main__":
    main()