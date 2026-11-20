import pandas as pd
import numpy as np
from scipy.optimize import linprog
from typing import Dict, Any
import json

# --------------------------------------------------------- 
# 1. THE DATASET (Pivot to Plant-Protein Isolates)
# ---------------------------------------------------------
# Note: Replace this with the Data Lead's CSV later.
ingredients_data = [
    # Name, Protein %, Fibre %, Cost (NGN/kg), Max %, Origin
    {"name": "Bambara Isolate (Plateau)", "protein_pct": 85.0, "fibre_pct": 2.0, "price_per_kg": 4500, "max_share": 0.50},
    {"name": "Bambara Isolate (Kano)", "protein_pct": 78.0, "fibre_pct": 2.5, "price_per_kg": 4200, "max_share": 0.50},
    {"name": "Cowpea Concentrate", "protein_pct": 60.0, "fibre_pct": 3.0, "price_per_kg": 2800, "max_share": 0.40},
    {"name": "Soy Isolate (Import)", "protein_pct": 90.0, "fibre_pct": 1.0, "price_per_kg": 5500, "max_share": 0.60},
    {"name": "Groundnut Concentrate", "protein_pct": 55.0, "fibre_pct": 4.0, "price_per_kg": 2500, "max_share": 0.30},
]
df = pd.DataFrame(ingredients_data)

# ---------------------------------------------------------
# 2. THE OPTIMISER FUNCTION (Feature 6)
# ---------------------------------------------------------
def optimize_blend(target_protein: float, max_fibre: float, batch_kg: float = 1000) -> Dict[str, Any]:
    num_ingredients = len(df)
    c = df['price_per_kg'].values
    
    A_ub = [
        df['fibre_pct'].values,      
        -df['protein_pct'].values    
    ]
    b_ub = [max_fibre, -target_protein]
    
    A_eq = [np.ones(num_ingredients)]
    b_eq = [1.0]
    
    bounds = [(0, row['max_share']) for _, row in df.iterrows()]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        blend_shares = result.x
        total_cost_per_kg = result.fun
        
        blend_output = []
        for i, row in df.iterrows():
            if blend_shares[i] > 0.001:
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
# 3. BLEND SIMULATION (Feature 5) - Adding Mentor's Warning
# ---------------------------------------------------------
def simulate_blend(blend_dict: Dict[str, float]) -> Dict[str, Any]:
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
        "cost_per_kg": round(total_cost, 2),
        "scientific_warning": "Nutrition and cost are exact weighted averages. Functional properties (e.g., WHC, gelation) may vary due to protein interactions and cannot be simply averaged."
    }

if __name__ == "__main__":
    print("--- TESTING NEW PLANT-PROTEIN OPTIMISER ---")
    # Request: 75% protein, max 2% fibre
    print(json.dumps(optimize_blend(target_protein=75.0, max_fibre=2.0), indent=2))