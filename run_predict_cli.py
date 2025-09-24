# run_predict_cli.py
import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "sales_model.pkl")
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "advertising.csv"),
    r"D:\Data_Science\Sales_prediction\advertising.csv"
]
CSV_PATH = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
if CSV_PATH is None:
    raise FileNotFoundError("advertising.csv not found. Place it in the project folder or update CSV_CANDIDATES.")

model = pickle.load(open(MODEL_PATH, "rb"))
df = pd.read_csv(CSV_PATH)

max_expected_sales = df["Sales"].max()
max_tv = df["TV"].max()
max_radio = df["Radio"].max()
max_newspaper = df["Newspaper"].max()

def get_float(prompt):
    while True:
        try:
            val = input(prompt).replace(",","").strip()
            return float(val)
        except ValueError:
            print("Please enter a valid number (no letters).")

tv_input = get_float("Enter TV advertising budget: ")
radio_input = get_float("Enter Radio advertising budget: ")
newspaper_input = get_float("Enter Newspaper advertising budget: ")

# Corrected scaling logic: if user input > dataset max -> cap to dataset max
tv_budget = min(max(tv_input, 0), max_tv)
radio_budget = min(max(radio_input, 0), max_radio)
newspaper_budget = min(max(newspaper_input, 0), max_newspaper)

user_input = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], columns=["TV", "Radio", "Newspaper"])
predicted_sales = float(model.predict(user_input)[0])

predicted_percentage = (predicted_sales / max_expected_sales) * 100
predicted_percentage = min(max(predicted_percentage, 0), 100)

print(f"\nPredicted Sales: {predicted_sales:.2f}")
print(f"Predicted Sales as percentage of max sales: {predicted_percentage:.2f}%")
