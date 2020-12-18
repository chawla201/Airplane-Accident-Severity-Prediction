import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("gb_classifier.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home_page.html")


@app.route("/predict", methods=["POST"])
def predict():
    feature_names = [
        "Safety_Score",
        "Days_Since_Inspection",
        "Total_Safety_Complaints",
        "Control_Metric",
        "Turbulence_In_gforces",
        "Cabin_Temperature",
        "Accident_Type_Code",
        "Max_Elevation",
        "Violations",
        "Adverse_Weather_Metric",
    ]
    features = pd.DataFrame(columns=feature_names)
    for feat in feature_names:
        features[feat] = [float(request.form.get(feat))]
    categorical = ["Violations", "Accident_Type_Code"]
    features[categorical] = features[categorical].astype("object")
    numerical = [
        "Safety_Score",
        "Days_Since_Inspection",
        "Total_Safety_Complaints",
        "Control_Metric",
        "Turbulence_In_gforces",
        "Cabin_Temperature",
        "Max_Elevation",
        "Adverse_Weather_Metric",
    ]
    X = features[numerical]
    violations = [
        "Violations_0",
        "Violations_1",
        "Violations_2",
        "Violations_3",
        "Violations_4",
        "Violations_5",
    ]
    accident_type_codes = [
        "Accident_Type_Code_1",
        "Accident_Type_Code_2",
        "Accident_Type_Code_3",
        "Accident_Type_Code_4",
        "Accident_Type_Code_5",
        "Accident_Type_Code_6",
        "Accident_Type_Code_7",
    ]
    for ele in violations:
        if int(features["Violations"][0]) == int(ele[-1:]):
            X[ele] = 1
        else:
            X[ele] = 0
    for ele in accident_type_codes:
        if int(features["Accident_Type_Code"][0]) == int(ele[-1:]):
            X[ele] = 1
        else:
            X[ele] = 0

    X = X.to_numpy()

    prediction = model.predict(X)
    output = pd.Series(prediction).map(
        {
            0: "Highly_Fatal_And_Damaging",
            1: "Minor_Damage_And_Injuries",
            2: "Significant_Damage_And_Fatalities",
            3: "Significant_Damage_And_Serious_Injuries",
        }
    )

    return render_template(
        "prediction.html",
        predicted_val=f"The Severity of the Accident is: {output[0]}",
    )


if __name__ == "__main__":
    app.run(debug=True)
