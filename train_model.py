import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "student_performance.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "student_model.pkl"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.pkl"
METRICS_PATH = MODEL_DIR / "metrics.pkl"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n==========================================")
print("   AI STUDENT PERFORMANCE PREDICTION")
print("          MODEL TRAINING")
print("==========================================\n")

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset loaded successfully!")
print(f"Total records: {len(df)}")


# ============================================================
# 3. CHECK DATA
# ============================================================

print("\nDataset columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "internal_score",
    "sleep_hours",
    "backlogs"
]

target = "final_score"

X = df[features]
y = df[target]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n==========================================")
print("           DATA SPLIT")
print("==========================================")

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ============================================================
# 6. CREATE RANDOM FOREST MODEL
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=14,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("Training model...\n")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 8. MAKE TEST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n==========================================")
print("          MODEL PERFORMANCE")
print("==========================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print(f"R² % : {r2 * 100:.2f}%")


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n==========================================")
print("          FEATURE IMPORTANCE")
print("==========================================")

for _, row in feature_importance.iterrows():

    print(
        f"{row['feature']:20s} "
        f"{row['importance'] * 100:.2f}%"
    )


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)

print("\nModel saved:")
print(MODEL_PATH)


# ============================================================
# 12. SAVE FEATURE IMPORTANCE
# ============================================================

joblib.dump(
    feature_importance,
    IMPORTANCE_PATH
)

print("Feature importance saved:")
print(IMPORTANCE_PATH)


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "r2_percentage": float(r2 * 100),
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

joblib.dump(metrics, METRICS_PATH)

print("Metrics saved:")
print(METRICS_PATH)


# ============================================================
# 14. SAMPLE PREDICTIONS
# ============================================================

print("\n==========================================")
print("          SAMPLE PREDICTIONS")
print("==========================================")

sample_data = X_test.head(5)

sample_predictions = model.predict(sample_data)

for i, prediction in enumerate(sample_predictions):

    actual = y_test.iloc[i]

    print(
        f"Actual: {actual:.1f} "
        f"| Predicted: {prediction:.1f}"
    )


print("\n==========================================")
print("       MODEL TRAINING SUCCESSFUL")
print("==========================================\n")