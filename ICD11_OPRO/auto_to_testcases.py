import json
import os

INTERACTIONS_FILE = "interactions.json"
TEST_CASES_FILE = "tests/test_cases.json"


def convert_to_test_cases():
    if not os.path.exists(INTERACTIONS_FILE):
        print("No interactions found.")
        return
    with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_cases = []
    for idx, entry in enumerate(data):
        if "user_input" in entry and "llm_response" in entry:
            test_case = {
                "id": f"user_case_{idx+1}",
                "question": entry["user_input"],
                "expected_aspects": [],
                "context": "",
                "category": "user",
                "crisis_level": "unknown"
            }
            if "user_feedback" in entry and entry["user_feedback"]:
                test_case["user_feedback"] = entry["user_feedback"]
            if "auto_evaluation" in entry and entry["auto_evaluation"]:
                test_case["auto_evaluation"] = entry["auto_evaluation"]
            test_cases.append(test_case)
    output = {"test_cases": test_cases}
    os.makedirs(os.path.dirname(TEST_CASES_FILE), exist_ok=True)
    with open(TEST_CASES_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(test_cases)} interactions to test cases.")

if __name__ == "__main__":
    convert_to_test_cases() 