import json
import os
from datetime import datetime

INTERACTIONS_FILE = "interactions.json"

def log_interaction(user_input, llm_response, user_feedback=None):
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "llm_response": llm_response,
        "user_feedback": user_feedback or {}
    }
    if os.path.exists(INTERACTIONS_FILE):
        with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(interaction)
    with open(INTERACTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Example usage
    log_interaction(
        user_input="I feel anxious and can't sleep at night.",
        llm_response="Thank you for sharing. It's important to acknowledge your feelings...",
        user_feedback={
            "satisfaction": 4,
            "empathy": 5,
            "accuracy": 4,
            "safety": 5,
            "comment": "Very helpful, I feel understood."
        }
    ) 