import json
import os
from evaluate_prompt import PromptEvaluator

INTERACTIONS_FILE = "interactions.json"
CONFIG_FILE = "config.json"

def auto_evaluate_interactions():
    if not os.path.exists(INTERACTIONS_FILE):
        print("No interactions found.")
        return
    with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    evaluator = PromptEvaluator(CONFIG_FILE)
    updated = False
    for entry in data:
        if "auto_evaluation" not in entry:
            result = evaluator.evaluate_prompt(entry["llm_response"], evaluation_method="fast")
            entry["auto_evaluation"] = {
                "overall_score": result.overall_score,
                "dimension_scores": result.dimension_scores,
                "timestamp": result.timestamp
            }
            updated = True
    if updated:
        with open(INTERACTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Auto evaluation completed and saved.")
    else:
        print("No new entries to evaluate.")

if __name__ == "__main__":
    auto_evaluate_interactions() 