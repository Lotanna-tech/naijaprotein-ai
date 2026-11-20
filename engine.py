import pandas as pd
import numpy as np
from scipy.optimize import linprog
from typing import Dict, Any
import json

# ---------------------------------------------------------
# 1. THE DATASET (From your guide's illustrative data)
# ---------------------------------------------------------
ingredients_data = [
    {"name": "Maize", "protein_pct": 9.0, "fibre_pct": 2.5, "price_per_kg": 500, "max_share": 0.60},
    {"name": "Soybean meal", "protein_pct": 45.0, "fibre_pct": 6.0, "price_per_kg": 900, "max_share": 0.40},
    {"name": "Groundnut cake", "protein_pct": 45.0, "fibre_pct": 7.0, "price_per_kg": 800, "max_share": 0.35},
    {"name": "Cowpea", "protein_pct": 23.0, "fibre_pct": 5.0, "price_per_kg": 700, "max_share": 0.30},
    {"name": "Wheat offal", "protein_pct": 15.0, "fibre_pct": 10.0, "price_per_kg": 350, "max_share": 0.30},
    {"name": "Palm kernel cake", "protein_pct": 18.0, "fibre_pct": 14.0, "price_per_kg": 300, "max_share": 0.25},
]
df = pd.DataFrame(ingredients_data)

# ---------------------------------------------------------
# 2. THE OPTIMISER FUNCTION (Feature 6)
# ---------------------------------------------------------
def optimize_blend(target_protein: float, max_fibre: float, batch_kg: float = 1000) -> Dict[str, Any]:
    num_ingredients = len(df)
    
    # Objective: Minimize cost (price per kg * share)
    c = df['price_per_kg'].values
    
    # Inequality constraints (A_ub * x <= b_ub)
    A_ub = [
        df['fibre_pct'].values,      # Fibre constraint
        -df['protein_pct'].values    # Protein constraint (negated)
    ]
    b_ub = [max_fibre, -target_protein]
    
    # Equality constraint (sum of shares = 1.0)
    A_eq = [np.ones(num_ingredients)]
    b_eq = [1.0]
    
    # Bounds (0 to max_share for each ingredient)
    bounds = [(0, row['max_share']) for _, row in df.iterrows()]
    
    # Run the HiGHS solver
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        blend_shares = result.x
        total_cost_per_kg = result.fun
        
        blend_output = []
        for i, row in df.iterrows():
            if blend_shares[i] > 0.001: # Filter out near-zero values
                blend_output.append({
                    "ingredient": row['name'],
                    "share_pct": round(blend_shares[i] * 100, 1),
                    "quantity_kg": round(blend_shares[i] * batch_kg, 1)
                })
                
        return {
            "status": "success",
            "blend": blend_output,
            "achieved_protein_pct": round(np.dot(blend_shares, df['protein_pct']), 2),
            "achieved_fibre_pct": round(np.dot(blend_shares, df['fibre_pct']), 2),
            "cost_per_kg": round(total_cost_per_kg, 2),
            "total_batch_cost": round(total_cost_per_kg * batch_kg, 2)
        }
    else:
        return {
            "status": "infeasible",
            "message": "Targets cannot be met with current ingredient limits.",
            "solver_detail": result.message
        }

# ---------------------------------------------------------
# 3. BLEND SIMULATION (Feature 5)
# ---------------------------------------------------------
def simulate_blend(blend_dict: Dict[str, float]) -> Dict[str, Any]:
    """Takes a dict of shares, e.g., {"Cowpea": 60, "Soybean meal": 30}"""
    if round(sum(blend_dict.values()), 2) != 100.0:
        return {"status": "error", "message": "Shares must sum to exactly 100%"}
        
    total_protein, total_fibre, total_cost = 0.0, 0.0, 0.0
    
    for name, share in blend_dict.items():
        row = df[df['name'] == name]
        if row.empty:
            return {"status": "error", "message": f"Ingredient '{name}' not found."}
            
        row = row.iloc[0]
        share_fraction = share / 100.0
        total_protein += row['protein_pct'] * share_fraction
        total_fibre += row['fibre_pct'] * share_fraction
        total_cost += row['price_per_kg'] * share_fraction
        
    return {
        "status": "success",
        "achieved_protein_pct": round(total_protein, 2),
        "achieved_fibre_pct": round(total_fibre, 2),
        "cost_per_kg": round(total_cost, 2)
    }

# ---------------------------------------------------------
# 4. TESTS (Only run when executing 'python engine.py')
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- REQUEST A: Feasible (20% Protein, 8% Fibre) ---")
    print(json.dumps(optimize_blend(target_protein=20.0, max_fibre=8.0), indent=2))
    
    print("\n--- REQUEST B: Infeasible (60% Protein, 8% Fibre) ---")
    print(json.dumps(optimize_blend(target_protein=60.0, max_fibre=8.0), indent=2))

    print("\n--- BLEND SIMULATION (Feature 5) ---")
    print(json.dumps(simulate_blend({"Cowpea": 60, "Soybean meal": 30, "Groundnut cake": 10}), indent=2))