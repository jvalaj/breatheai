Overview
This project aims to detect cough sounds from an audio file using machine learning. The model is trained on a dataset containing metadata and 31 audio features extracted from 3029 rows of data sourced from Kaggle. The model employs a Gradient Boosting classifier to predict the likelihood of a cough being detected in the given audio input.

Features
Audio Processing: The system processes audio files (in various formats such as .m4a, .obb, .webm, .mp3, .flac) and extracts relevant audio features for prediction.
Feature Extraction: Extracts 31 features, including Mel-frequency cepstral coefficients (MFCCs), chroma, and spectral contrast, using the librosa library.
Machine Learning Model: Uses a Gradient Boosting classifier (gbm_model.pkl), trained on a Kaggle dataset with metadata and audio features.
Prediction: After feature extraction, the model predicts the likelihood of a cough being present in the given audio file.
Scalability: The system works on individual audio files and can be used for batch processing by calling the function multiple times.
Requirements
Before running the project, make sure you have the following dependencies installed:

Python 3.x
Libraries:
librosa: For audio processing
numpy: For handling arrays and numerical operations
pydub: For audio file format conversion
scikit-learn: For machine learning and model prediction
pickle: For saving and loading the trained model
ffmpeg: For audio file conversion
pandas: For handling data frames (if needed for preprocessing)
Install the dependencies using the following:

bash
Copy
Edit
pip install librosa numpy pydub scikit-learn pickle ffmpeg pandas
Dataset
The model was trained on a dataset from Kaggle, consisting of 3029 rows of metadata and 31 audio features. These features were extracted from raw audio files representing a variety of sounds, including coughs.

Data Features:
Mel-frequency cepstral coefficients (MFCCs)
Chroma spectral features
Spectral contrast
Usage
Prepare the Audio File: Place the audio file you want to process in a directory. Supported file formats include .m4a, .obb, .webm, .mp3, and .flac.

Run the Script: Replace the file path in the audio_file variable with the path to your audio file and run the script.

python
Copy
Edit
audio_file = '/path/to/your/audio/file.m4a'  # Path to your audio file
predict_from_audio(audio_file)
Output: The script will print the prediction result, showing the likelihood of a cough being detected in the audio file.
bash
Copy
Edit
Cough Detected Prediction: 92.5%
Files in the Project
predict_cough.py: Main script for extracting features, loading the model, and making predictions on a given audio file.
gbm_model.pkl: The trained Gradient Boosting model.
scaler.pkl: The scaler used to scale features before making predictions.
Model Explanation
The model used is a Gradient Boosting Machine (GBM), a type of ensemble machine learning model that builds multiple weak learners (decision trees) and combines them to make accurate predictions. The GBM model was trained on a dataset that includes audio features such as MFCCs, chroma, and spectral contrast to distinguish between normal sounds and coughs.

Troubleshooting
Error with audio file format: If the audio file is not in WAV format, it will be automatically converted using the pydub library. Make sure ffmpeg is installed and accessible in your system's PATH.

Missing Dependencies: If any libraries are missing, install them using the pip install command mentioned above.

Incorrect predictions: Ensure that the dataset is representative of the data you're testing, as the model may not perform well on data with a different distribution.

Future Enhancements
Improve model accuracy by retraining with more diverse audio data.
Add support for batch processing of multiple audio files.
Implement a web or mobile interface for real-time cough detection.
License
This project is open-source. Feel free to fork and modify it for personal or academic use.

Contact
If you have any questions or need further clarification, feel free to reach out.
