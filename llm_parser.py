import boto3
import json
import re

# --- FLIP THIS TO FALSE WHEN AWS KEYS ARRIVE ---
USE_MOCK = True 

# Initialize client (Will fail gracefully if no AWS keys exist yet)
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

def parse_natural_language(user_text: str) -> dict:
    """Extracts protein and fibre targets from a user's sentence."""
    if USE_MOCK:
        print("   [MOCK MODE]: Simulating Bedrock (Parsing)...")
        # Smarter regex to catch "20% protein" and "fiber under 8%"
        protein_match = re.search(r'(\d+)\s*%?\s*(?:protein)', user_text, re.IGNORECASE)
        fibre_match = re.search(r'(?:fibre|fiber)\s*(?:under|below|max|max\s*)?\s*(\d+)\s*%?', user_text, re.IGNORECASE)
        
        # Fallback if regex misses
        if not fibre_match:
            fibre_match = re.search(r'(\d+)\s*%\s*(?:fibre|fiber)', user_text, re.IGNORECASE)

        return {
            "target_protein": float(protein_match.group(1)) if protein_match else 15.0,
            "max_fibre": float(fibre_match.group(1)) if fibre_match else 10.0
        }

    # --- REAL BEDROCK CODE ---
    prompt = f"""
    You are a formulation AI. Extract the protein target and maximum fibre from the user's request.
    Respond ONLY with a valid JSON object using these keys: "target_protein" and "max_fibre".
    If a value is not mentioned, default to target_protein: 15, max_fibre: 10.
    
    User request: "{user_text}"
    """
    
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"max_new_tokens": 100, "temperature": 0.0}
    })
    
    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0", 
            contentType="application/json",
            accept="application/json",
            body=body
        )
        response_body = json.loads(response.get('body').read())
        result_text = response_body['output']['message']['content'][0]['text']
        
        # Clean markdown if LLM adds it
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        return json.loads(result_text)
        
    except Exception as e:
        return {"error": str(e), "target_protein": 15.0, "max_fibre": 10.0}


def explain_blend(blend_result: dict) -> str:
    """Takes the optimiser JSON and returns a plain-English explanation."""
    
    # 1. Handle Infeasible cases immediately
    if blend_result.get("status") == "infeasible":
        return "This target is mathematically impossible with the current ingredients. You would need to add a higher-protein crop or relax your fibre limit."
        
    # 2. Extract ingredients for the mock explanation
    used_ingredients = [f"{b['share_pct']}% {b['ingredient']}" for b in blend_result["blend"]]
    blend_summary = ", ".join(used_ingredients)
    
    if USE_MOCK:
        print("   [MOCK MODE]: Simulating Bedrock (Explanation)...")
        return f"To achieve {blend_result['achieved_protein_pct']}% protein while keeping fibre at {blend_result['achieved_fibre_pct']}%, the optimal blend is {blend_summary}. The total cost is {blend_result['total_batch_cost']:,.2f} NGN. The optimiser mathematically selected the most cost-effective isolates to hit your protein targets while respecting the maximum inclusion limits."

    # --- REAL BEDROCK CODE ---
    prompt = f"""
    You are a food science AI. Explain why this optimiser chose this blend in 2 short sentences.
    Blend Data: {json.dumps(blend_result)}
    Keep it professional and easy for a factory manager to understand.
    """
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"max_new_tokens": 100, "temperature": 0.2}
    })
    
    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0", 
            contentType="application/json",
            accept="application/json",
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return response_body['output']['message']['content'][0]['text'].strip()
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"