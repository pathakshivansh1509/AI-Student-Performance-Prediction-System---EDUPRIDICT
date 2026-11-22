
# =========================================================
# EDUPREDICT AI - STUDENT PERFORMANCE PREDICTION SYSTEM
# =========================================================

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

app = Flask(__name__)

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "student_model.pkl"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.pkl"
METRICS_PATH = MODEL_DIR / "metrics.pkl"
HISTORY_PATH = MODEL_DIR / "prediction_history.json"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "internal_score",
    "sleep_hours",
    "backlogs"
]

# =========================================================
# LOAD MODEL
# =========================================================

model = None
feature_importance = []
metrics = {}

try:
    model = joblib.load(MODEL_PATH)
    feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)
    metrics = joblib.load(METRICS_PATH)

    print("=" * 45)
    print("EDUPREDICT MODEL LOADED SUCCESSFULLY")
    print("Model:", MODEL_PATH.name)
    print("Metrics:", METRICS_PATH.name)
    print("=" * 45)

except Exception as e:
    print("MODEL LOADING ERROR:", str(e))


# =========================================================
# FEATURE IMPORTANCE FORMATTER
# =========================================================

def get_feature_importance():
    """Convert feature importance into template-friendly dictionaries."""

    result = []

    try:
        raw = feature_importance

        if isinstance(raw, dict):
            for feature, importance in raw.items():
                result.append({
                    "feature": str(feature),
                    "importance": float(importance)
                })

        elif isinstance(raw, pd.DataFrame):
            for _, row in raw.iterrows():
                result.append({
                    "feature": str(row["feature"]),
                    "importance": float(row["importance"])
                })

        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    result.append({
                        "feature": str(item.get("feature", "Unknown")),
                        "importance": float(item.get("importance", 0))
                    })
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    result.append({
                        "feature": str(item[0]),
                        "importance": float(item[1])
                    })

        elif isinstance(raw, np.ndarray):
            values = raw.flatten()

            if len(values) == len(FEATURES):
                result = [
                    {
                        "feature": feature,
                        "importance": float(value)
                    }
                    for feature, value in zip(FEATURES, values)
                ]

        # If saved importance is empty, try extracting it from the model.
        if not result and model is not None:
            estimator = model

            if hasattr(estimator, "named_steps"):
                for step in reversed(list(estimator.named_steps.values())):
                    if hasattr(step, "feature_importances_"):
                        estimator = step
                        break

            if hasattr(estimator, "feature_importances_"):
                values = estimator.feature_importances_

                if len(values) == len(FEATURES):
                    result = [
                        {
                            "feature": feature,
                            "importance": float(value)
                        }
                        for feature, value in zip(FEATURES, values)
                    ]

        result = [
            item for item in result
            if np.isfinite(item["importance"])
        ]

        result.sort(
            key=lambda item: item["importance"],
            reverse=True
        )

    except Exception as e:
        print("FEATURE IMPORTANCE ERROR:", str(e))

    return result


# =========================================================
# HISTORY FUNCTIONS
# =========================================================

def load_history():
    try:
        if not HISTORY_PATH.exists():
            return []

        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            data = data.get("history", [])

        return data if isinstance(data, list) else []

    except Exception as e:
        print("HISTORY LOADING ERROR:", str(e))
        return []


def save_history(history):
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        with open(HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump({"history": history}, file, indent=4)

        return True

    except Exception as e:
        print("HISTORY SAVING ERROR:", str(e))
        return False


def add_prediction_to_history(record):
    history = load_history()
    history.insert(0, record)
    return save_history(history[:50])


# =========================================================
# PERFORMANCE CATEGORY
# =========================================================

def get_category(score):
    if score < 50:
        return "POOR"
    elif score < 65:
        return "AVERAGE"
    elif score < 80:
        return "GOOD"
    return "EXCELLENT"


# =========================================================
# AI INSIGHT
# =========================================================

def get_ai_insight(category):
    insights = {
        "POOR": (
            "Consider improving study consistency, attendance "
            "and academic preparation."
        ),
        "AVERAGE": (
            "Consistent study habits and stronger preparation "
            "may help improve future performance."
        ),
        "GOOD": (
            "Maintain your study routine and attendance "
            "to support further improvement."
        ),
        "EXCELLENT": (
            "Keep maintaining consistent study habits "
            "and academic preparation."
        )
    }

    return insights.get(category, "Review the student's academic progress.")


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    formatted_feature_importance = get_feature_importance()

    print("Feature importance count:", len(formatted_feature_importance))

    return render_template(
        "index.html",
        metrics=metrics,
        feature_importance=formatted_feature_importance
    )


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "ML model is not loaded. Check the model files."
            }), 500

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "error": "No input data received."
            }), 400

        values = {}

        limits = {
            "study_hours": (0, 24),
            "attendance": (0, 100),
            "previous_score": (0, 100),
            "assignment_score": (0, 100),
            "internal_score": (0, 100),
            "sleep_hours": (0, 24),
            "backlogs": (0, 20)
        }

        for name in FEATURES:
            value = float(data.get(name, 0))

            if not np.isfinite(value):
                return jsonify({
                    "success": False,
                    "error": f"{name} must be a valid number."
                }), 400

            low, high = limits[name]

            if not low <= value <= high:
                return jsonify({
                    "success": False,
                    "error": f"{name} must be between {low} and {high}."
                }), 400

            values[name] = value

        input_data = pd.DataFrame(
            [[values[name] for name in FEATURES]],
            columns=FEATURES
        )

        prediction = float(model.predict(input_data)[0])
        prediction = round(max(0, min(100, prediction)), 1)

        category = get_category(prediction)
        insight = get_ai_insight(category)
        timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

        history_record = {
            "prediction": prediction,
            "category": category,
            **values,
            "timestamp": timestamp
        }

        history_saved = add_prediction_to_history(history_record)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "category": category,
            "model_r2": float(metrics.get("r2_percentage", 0)),
            "mae": float(metrics.get("mae", 0)),
            "rmse": float(metrics.get("rmse", 0)),
            "insight": insight,
            "feature_importance": get_feature_importance(),
            "history_saved": history_saved
        })

    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Please enter valid numeric values."
        }), 400

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        return jsonify({
            "success": False,
            "error": "An error occurred while generating the prediction."
        }), 500


# =========================================================
# HISTORY API
# =========================================================

@app.route("/history", methods=["GET"])
def history():
    return jsonify({
        "success": True,
        "history": load_history()
    })


# =========================================================
# CLEAR HISTORY
# =========================================================

@app.route("/history/clear", methods=["POST"])
def clear_history():
    if save_history([]):
        return jsonify({
            "success": True,
            "message": "Prediction history cleared successfully."
        })

    return jsonify({
        "success": False,
        "error": "Unable to clear prediction history."
    }), 500


# =========================================================
# ANALYTICS API
# =========================================================

@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        history_data = load_history()

        predictions = [
            float(item.get("prediction", 0))
            for item in history_data
        ]

        predictions = [
            value for value in predictions
            if np.isfinite(value)
        ]

        total = len(predictions)

        categories = {
            "POOR": 0,
            "AVERAGE": 0,
            "GOOD": 0,
            "EXCELLENT": 0
        }

        for item in history_data:
            category = str(
                item.get("category", "AVERAGE")
            ).upper()

            if category in categories:
                categories[category] += 1

        prediction_trend = []

        for index, item in enumerate(reversed(history_data), start=1):
            prediction_trend.append({
                "index": index,
                "prediction": float(item.get("prediction", 0)),
                "category": item.get("category", "AVERAGE"),
                "timestamp": item.get("timestamp", "")
            })

        study_vs_prediction = []
        attendance_vs_prediction = []
        previous_score_vs_prediction = []

        for item in history_data:
            prediction = float(item.get("prediction", 0))

            study_vs_prediction.append({
                "study_hours": float(item.get("study_hours", 0)),
                "prediction": prediction
            })

            attendance_vs_prediction.append({
                "attendance": float(item.get("attendance", 0)),
                "prediction": prediction
            })

            previous_score_vs_prediction.append({
                "previous_score": float(item.get("previous_score", 0)),
                "prediction": prediction
            })

        return jsonify({
            "success": True,
            "total_predictions": total,
            "average_prediction": round(sum(predictions) / total, 2) if total else 0,
            "highest_prediction": max(predictions) if total else 0,
            "lowest_prediction": min(predictions) if total else 0,
            "categories": categories,
            "prediction_trend": prediction_trend,
            "study_vs_prediction": study_vs_prediction,
            "attendance_vs_prediction": attendance_vs_prediction,
            "previous_score_vs_prediction": previous_score_vs_prediction
        })

    except Exception as e:
        print("ANALYTICS ERROR:", str(e))
        return jsonify({
            "success": False,
            "error": "Unable to generate analytics."
        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "algorithm": "Random Forest",
        "r2": float(metrics.get("r2_percentage", 0))
    })


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    print("=" * 45)
    print("EDUPREDICT AI SYSTEM")
    print("Server: http://127.0.0.1:5000")
    print("Health: http://127.0.0.1:5000/health")
    print("Analytics: http://127.0.0.1:5000/analytics")
    print("=" * 45)

    app.run(debug=True)
