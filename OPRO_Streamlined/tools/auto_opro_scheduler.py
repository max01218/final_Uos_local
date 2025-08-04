import os
import time
import json
from run_opro import OPROInterface

INTERACTIONS_FILE = "interactions.json"
OPTIMIZED_PROMPT_FILE = "prompts/optimized_prompt.txt"
LAST_OPTIMIZED_FILE = "last_optimized.json"
CHECK_INTERVAL_SECONDS = 60 * 60 * 24  # 24 hours
MIN_NEW_INTERACTIONS = 10


def get_last_optimized_count():
    if os.path.exists(LAST_OPTIMIZED_FILE):
        with open(LAST_OPTIMIZED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("last_count", 0)
    return 0

def set_last_optimized_count(count):
    with open(LAST_OPTIMIZED_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_count": count}, f)

def get_interaction_count():
    if os.path.exists(INTERACTIONS_FILE):
        with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return len(data)
    return 0

def auto_opro_loop():
    while True:
        last_count = get_last_optimized_count()
        current_count = get_interaction_count()
        print(f"Current interactions: {current_count}, Last optimized: {last_count}")
        if current_count - last_count >= MIN_NEW_INTERACTIONS:
            print("Triggering OPRO optimization...")
            interface = OPROInterface()
            interface.run_opro_optimization()
            set_last_optimized_count(current_count)
            print("OPRO optimization completed and prompt deployed.")
        else:
            print(f"Not enough new interactions. Waiting...")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    auto_opro_loop() 