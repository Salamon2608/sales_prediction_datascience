# app.py
import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, jsonify

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "sales_model.pkl")
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "advertising.csv"),
    r"D:\Data_Science\Sales_prediction\advertising.csv"
]
CSV_PATH = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
if CSV_PATH is None:
    raise FileNotFoundError("advertising.csv not found. Place it in the project folder or update CSV_CANDIDATES.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("sales_model.pkl not found. Run train_model.py first.")

# load model and dataset
model = pickle.load(open(MODEL_PATH, "rb"))
df = pd.read_csv(CSV_PATH)
max_expected_sales = df["Sales"].max()
max_tv = df["TV"].max()
max_radio = df["Radio"].max()
max_newspaper = df["Newspaper"].max()

app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    # render page normally; template contains JS to call /predict
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # if request has JSON (AJAX), read JSON; else read regular form
    data = None
    if request.is_json:
        data = request.get_json()
        try:
            tv = float(data.get("tv", 0))
            radio = float(data.get("radio", 0))
            newspaper = float(data.get("newspaper", 0))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid numeric input."}), 400
    else:
        # legacy form submit (keeps compatibility)
        try:
            tv = float(request.form.get("tv", 0))
            radio = float(request.form.get("radio", 0))
            newspaper = float(request.form.get("newspaper", 0))
        except (ValueError, TypeError):
            return render_template("index.html", prediction_text="Please enter valid numbers.")

    # cap inputs (keep them in training range)
    tv = min(max(tv, 0), max_tv)
    radio = min(max(radio, 0), max_radio)
    newspaper = min(max(newspaper, 0), max_newspaper)

    user_input = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"])
    predicted_sales = float(model.predict(user_input)[0])
    predicted_percentage = (predicted_sales / max_expected_sales) * 100
    predicted_percentage = min(max(predicted_percentage, 0), 100)

    result_text = f"Predicted Sales: {predicted_sales:.2f} — ({predicted_percentage:.2f}% of max)"

    # If AJAX/JSON, return JSON so front-end JS can display it without reload
    if request.is_json:
        return jsonify({
            "prediction_text": result_text,
            "predicted_sales": predicted_sales,
            "predicted_percentage": round(predicted_percentage, 2)
        })

    # Otherwise render the template (fallback)
    return render_template("index.html", prediction_text=result_text)


if __name__ == "__main__":
    app.run(debug=True)
