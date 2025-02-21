import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib  # For saving the model

# Load the dataset
df = pd.read_csv("finalData.csv")  # Replace with actual file name

# Separate features (X) and target variable (y)
X = df.iloc[:, 2:34]  # Selecting only feature columns
y = df["cough_detected"]

# Scale the features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting model (since it's the best)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
score = mean_squared_error(y_test, y_pred) ** 0.5  # RMSE without squared=False

print(f"Gradient Boosting RMSE: {score}")

# Save the model and scaler
joblib.dump(model, 'gbm.pkl')  # Save the model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
print("Model and scaler saved as 'gbm.pkl' and 'scaler.pkl'")
