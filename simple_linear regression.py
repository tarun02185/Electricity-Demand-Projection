import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_excel(r"C:\Users\tarun\OneDrive\Desktop\data.xlsx", engine="openpyxl")

# Drop rows with missing values
data = data.dropna()

# Convert timestamp to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract time-based features
data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month
data['year'] = data['timestamp'].dt.year

# Create lag features
data['prev_demand'] = data['electricity_demand'].shift(1)
data['rolling_avg'] = data['electricity_demand'].rolling(window=3).mean()

# Fill NaN values from shift/rolling
data = data.bfill()

# Drop timestamp column
data = data.drop(columns=['timestamp'])

# Define features and target
X = data.drop(columns=['electricity_demand'])
y = data['electricity_demand']

# Split into train, dev, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
model_path = r"C:\Users\tarun\OneDrive\Desktop\linear_regression_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")

# Load the model
model = joblib.load(model_path)
print("Model loaded successfully.")

# Make predictions
y_train_pred = model.predict(X_train)
y_dev_pred = model.predict(X_dev)
y_test_pred = model.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    accuracy = r2 * 100
    print(f'{dataset_name} Set Evaluation:')
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R^2 Score: {r2:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
    print('-' * 40)

# Evaluate
evaluate_model(y_train, y_train_pred, "Train")
evaluate_model(y_dev, y_dev_pred, "Dev")
evaluate_model(y_test, y_test_pred, "Test")
