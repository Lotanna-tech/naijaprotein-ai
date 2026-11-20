from engine import optimize_blend
from llm_parser import parse_natural_language, explain_blend # Import the new function
import json

def run_ai_pipeline(user_text: str):
    print(f"👤 User: '{user_text}'\n")
    
    print("🧠 Calling AI to extract targets...")
    extracted_targets = parse_natural_language(user_text)
    print(f"   Extracted: {extracted_targets}\n")
    
    print("📊 Running Linear Programming Optimiser...")
    blend_result = optimize_blend(
        target_protein=extracted_targets.get("target_protein", 15.0),
        max_fibre=extracted_targets.get("max_fibre", 10.0)
    )
    
    print("\n✅ Final Result:")
    print(json.dumps(blend_result, indent=2))
    
    # 3. AI explains the result
    print("\n🗣️ AI Explanation:")
    explanation = explain_blend(blend_result)
    print(explanation)

if __name__ == "__main__":
    prompt = "I need to make a cheap mix for plant-based mince with at least 20% protein and keep the fiber under 8%."
    run_ai_pipeline(prompt)