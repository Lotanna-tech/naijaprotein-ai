import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import json

# ---------------------------------------------------------
# 1. GENERATE SYNTHETIC DATA (Since we don't have real lab data)
# ---------------------------------------------------------
# We create a dataset of 100 fake crops. 
# Water Holding Capacity is roughly Protein * 0.1 + Fiber * 0.4 + some noise.
np.random.seed(42)
n_samples = 100
protein = np.random.uniform(5, 45, n_samples)
fat = np.random.uniform(1, 20, n_samples)
fibre = np.random.uniform(1, 15, n_samples)

# Synthetic formula: Water holding = (0.5 * protein) + (1.2 * fibre) - (0.1 * fat) + noise
water_holding = (0.5 * protein) + (1.2 * fibre) - (0.1 * fat) + np.random.normal(0, 0.5, n_samples)

df = pd.DataFrame({
    "protein_pct": protein,
    "fat_pct": fat,
    "fibre_pct": fibre,
    "water_holding": water_holding
})

# ---------------------------------------------------------
# 2. TRAIN THE MACHINE LEARNING MODEL
# ---------------------------------------------------------
# We use a Ridge Regression model (good for small datasets with correlated features)
X = df[["protein_pct", "fat_pct", "fibre_pct"]]
y = df["water_holding"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Calculate the error on unseen data (Mean Absolute Error)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Calculate the baseline error (just guessing the average every time)
baseline_preds = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_preds)

# ---------------------------------------------------------
# 3. THE PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_water_holding(protein_pct: float, fat_pct: float, fibre_pct: float) -> dict:
    """
    Predicts the water holding capacity of a new ingredient based on its composition.
    """
    # Create a dataframe for the single input
    input_data = pd.DataFrame([{
        "protein_pct": protein_pct,
        "fat_pct": fat_pct,
        "fibre_pct": fibre_pct
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    return {
        "status": "success",
        "predicted_water_holding": round(prediction, 2),
        "model_mae": round(mae, 3),
        "baseline_mae": round(baseline_mae, 3),
        "confidence": "Proof of Concept (Synthetic Data)"
    }

if __name__ == "__main__":
    print("--- TESTING ML PREDICTION ---")
    # Test it with a hypothetical crop: 25% protein, 5% fat, 8% fibre
    result = predict_water_holding(protein_pct=25.0, fat_pct=5.0, fibre_pct=8.0)
    print(json.dumps(result, indent=2))