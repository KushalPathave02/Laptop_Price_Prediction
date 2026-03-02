import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "laptop_model.pkl")
DF_PATH = os.path.join("model", "df.pkl")


def _load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DF_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python preprocess.py` after placing dataset at data/laptop_data.csv"
        )
    model = pickle.load(open(MODEL_PATH, "rb"))
    df = pickle.load(open(DF_PATH, "rb"))
    return model, df


def _make_features(payload: dict) -> pd.DataFrame:
    # Expected keys
    required = [
        "company",
        "type",
        "ram",
        "weight",
        "touchscreen",
        "ips",
        "cpu",
        "gpu",
        "os",
    ]
    missing_keys = [k for k in required if k not in payload]
    if missing_keys:
        raise ValueError(f"Missing fields: {', '.join(missing_keys)}")

    empty_values = [k for k in required if payload.get(k) is None or str(payload.get(k)).strip() == ""]
    if empty_values:
        raise ValueError(f"Empty fields: {', '.join(empty_values)}")

    row = {
        "Company": str(payload["company"]),
        "TypeName": str(payload["type"]),
        "Ram": int(payload["ram"]),
        "Weight": float(payload["weight"]),
        "Touchscreen": int(payload["touchscreen"]),
        "IPS": int(payload["ips"]),
        "Cpu Brand": str(payload["cpu"]),
        "Gpu Brand": str(payload["gpu"]),
        "OpSys": str(payload["os"]),
    }

    return pd.DataFrame([row])


@app.route("/")
def index():
    try:
        _, df = _load_artifacts()
        return render_template(
            "index.html",
            companies=sorted(df["Company"].unique()),
            types=sorted(df["TypeName"].unique()),
            cpu=sorted(df["Cpu Brand"].unique()),
            gpu=sorted(df["Gpu Brand"].unique()),
            os=sorted(df["OpSys"].unique()),
        )
    except Exception as e:
        return (
            render_template("index.html", error_text=str(e)),
            500,
        )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, df = _load_artifacts()

        payload = {
            "company": request.form.get("company"),
            "type": request.form.get("type"),
            "ram": request.form.get("ram"),
            "weight": request.form.get("weight"),
            "touchscreen": request.form.get("touchscreen"),
            "ips": request.form.get("ips"),
            "cpu": request.form.get("cpu"),
            "gpu": request.form.get("gpu"),
            "os": request.form.get("os"),
        }

        X = _make_features(payload)
        pred_log = float(model.predict(X)[0])
        prediction = int(np.exp(pred_log))

        return render_template(
            "index.html",
            prediction_text=f"Estimated Price: ₹ {prediction}",
            companies=sorted(df["Company"].unique()),
            types=sorted(df["TypeName"].unique()),
            cpu=sorted(df["Cpu Brand"].unique()),
            gpu=sorted(df["Gpu Brand"].unique()),
            os=sorted(df["OpSys"].unique()),
        )
    except Exception as e:
        try:
            _, df = _load_artifacts()
            return (
                render_template(
                    "index.html",
                    error_text=str(e),
                    companies=sorted(df["Company"].unique()),
                    types=sorted(df["TypeName"].unique()),
                    cpu=sorted(df["Cpu Brand"].unique()),
                    gpu=sorted(df["Gpu Brand"].unique()),
                    os=sorted(df["OpSys"].unique()),
                ),
                400,
            )
        except Exception:
            return (render_template("index.html", error_text=str(e)), 400)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        model, _ = _load_artifacts()
        payload = request.get_json(force=True)
        X = _make_features(payload)
        pred_log = float(model.predict(X)[0])
        prediction = float(np.exp(pred_log))
        return jsonify({"predicted_price_inr": int(round(prediction))})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
