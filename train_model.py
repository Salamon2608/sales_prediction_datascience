# train_model.py
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(__file__)
CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "advertising.csv"),
    r"D:\Data_Science\Sales_prediction\advertising.csv"  # fallback if you didn't copy file
]

csv_path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(
        "advertising.csv not found. Put advertising.csv in the project folder or update the path in train_model.py."
    )

print("Loading dataset from:", csv_path)
df = pd.read_csv(csv_path)

print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Handle missing values
print("\nMissing values per column:")
print(df.isnull().sum())
df = df.fillna(df.mean())

# EDA (plots) - these open windows; remove if running on headless machine
try:
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(df)
    plt.show()
except Exception as e:
    print("Plotting skipped (might be headless). Error:", e)

# Features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Predictions & evaluation
y_pred = model.predict(X_test)
comparison = pd.DataFrame({"Actual Sales": y_test.reset_index(drop=True), "Predicted Sales": y_pred})
print("\nActual vs Predicted (first 5):")
print(comparison.head())

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Save the trained model
model_path = os.path.join(BASE_DIR, "sales_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print("\nModel saved as:", model_path)
