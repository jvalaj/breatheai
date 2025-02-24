import os
import librosa
import numpy as np
import pickle  # For loading the model and scaler
from pydub import AudioSegment
import tempfile

def extract_features(file_path):
    try:
        # Convert audio file to WAV format if necessary
        if file_path.endswith(('.obb', '.webm', '.mp3', '.flac')):
            audio = AudioSegment.from_file(file_path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                file_path = tmp_file.name
                audio.export(file_path, format='wav')

        # Load audio file
        y, sr = librosa.load(file_path)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Combine features into a single array
        features = np.hstack((np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1)))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_from_audio(audio_file_path):
    # Extract features from the audio file
    features = extract_features(audio_file_path)

    if features is not None:
        # Load the pre-trained model and scaler
        with open('gbm_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Prepare the features for prediction (scaling)
        features_scaled = scaler.transform([features])

        # Make prediction using the loaded model
        prediction = model.predict(features_scaled)

        # Print the result
        print(f"Cough Detected Prediction: {prediction[0] * 100}%")

# Example usage
audio_file = 'your path here'  # Path to your audio file
predict_from_audio(audio_file)