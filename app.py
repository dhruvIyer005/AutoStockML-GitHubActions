import math
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("model.joblib")


@app.get("/")
def home():
    return jsonify({
        "project": "AutoStockML",
        "status": "running",
        "health": "/health",
        "prediction": "/predict",
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    prices = data.get("prices")

    if not isinstance(prices, list) or len(prices) != 5:
        return jsonify({
            "error": "Provide exactly five prices"
        }), 400

    try:
        values = [float(price) for price in prices]
    except (ValueError, TypeError):
        return jsonify({"error": "Prices must be numeric"}), 400

    if not all(math.isfinite(p) and p > 0 for p in values):
        return jsonify({
            "error": "Prices must be finite positive numbers"
        }), 400

    result = float(model.predict([values])[0])

    return jsonify({
        "model": "LinearRegression",
        "predicted_next_close": round(result, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)