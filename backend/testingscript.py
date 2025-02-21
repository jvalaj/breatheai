import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV files
metadata = pd.read_csv("metadata.csv")  
audio_features = pd.read_csv("audio.csv")  

# Merge on uuid
df = metadata.merge(audio_features, on="uuid")

# Select Features and Target
X = df.drop(columns=["uuid", "datetime", "cough_detected"])  # Features
y = df["cough_detected"]  # Target variable

# Encode categorical variables (OneHotEncoder for 'gender' and 'status')
X = pd.get_dummies(X, columns=['gender', 'status'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
