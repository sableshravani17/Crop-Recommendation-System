from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset and model
df = pd.read_csv("crop_recommendation.csv")
model = joblib.load("crop_model.pkl")

# -------------------------------
# REGION-WISE CATEGORY MAPPING
# -------------------------------
region_map = {
    "North India": ["apple", "grapes", "kidneybeans", "maize", "lentil", "pomegranate"],
    "South India": ["banana", "coconut", "coffee", "blackgram", "mungbean", "mothbeans"],
    "East India": ["jute", "rice", "mango", "papaya", "pigeonpeas"],
    "West India": ["cotton", "chickpea", "muskmelon", "orange", "watermelon"]
}

# Reverse map: crop → region
crop_to_region = {}
for region, crops in region_map.items():
    for crop in crops:
        crop_to_region[crop] = region

# Feature Description
feature_description = {
    "N": "Nitrogen level in soil (mg/kg)",
    "P": "Phosphorus level in soil (mg/kg)",
    "K": "Potassium level in soil (mg/kg)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH value",
    "rainfall": "Rainfall (mm)"
}

# Dataset-based Ranges
feature_ranges = {
    "N": (df["N"].min(), df["N"].max()),
    "P": (df["P"].min(), df["P"].max()),
    "K": (df["K"].min(), df["K"].max()),
    "temperature": (df["temperature"].min(), df["temperature"].max()),
    "humidity": (df["humidity"].min(), df["humidity"].max()),
    "ph": (df["ph"].min(), df["ph"].max()),
    "rainfall": (df["rainfall"].min(), df["rainfall"].max()),
}


@app.route("/")
def index():
    return render_template("index.html",
                           region_map=region_map,
                           feature_description=feature_description,
                           feature_ranges=feature_ranges,
                           error_message=None)


@app.route("/predict", methods=["POST"])
def predict():
    user_input = {}

    for feature in feature_description.keys():
        value = float(request.form[feature])
        min_val, max_val = feature_ranges[feature]

        if not (min_val <= value <= max_val):
            # Show same page with error message
            return render_template(
                "index.html",
                region_map=region_map,
                feature_description=feature_description,
                feature_ranges=feature_ranges,
                error_message=f"Invalid input for '{feature}'. Allowed range: {min_val} - {max_val}"
            )

        user_input[feature] = value

    # Prediction
    features = [[
        user_input["N"], user_input["P"], user_input["K"],
        user_input["temperature"], user_input["humidity"],
        user_input["ph"], user_input["rainfall"]
    ]]

    prediction = model.predict(features)[0]
    region = crop_to_region.get(prediction, "Unknown Region")

    return render_template("result.html", crop=prediction, region=region)


if __name__ == "__main__":
    app.run(debug=True)
