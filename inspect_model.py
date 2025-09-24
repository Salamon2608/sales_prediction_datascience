# inspect_model.py
import os
import pickle

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "sales_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("sales_model.pkl not found. Run train_model.py first.")

model = pickle.load(open(MODEL_PATH, "rb"))
print("Model type:", type(model))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

features = ["TV", "Radio", "Newspaper"]
for feat, coef in zip(features, model.coef_):
    print(f"{feat}: {coef}")
