#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch ICD-11 Chapter 6 (Mental, behavioural or neuro-developmental disorders)
Outputs:
    icd11_ch6_entities.csv   # Main fields
    raw/???.json             # Complete JSON for each code for traceability
"""

import os, json, csv, requests, pathlib, time
from typing import Dict, List, Set
from urllib.parse import urlparse

# ------------------------------------------------------------------
# 1. Authentication data (use environment variables or read file)
CLIENT_ID     = os.getenv("ICD_ID")      # export ICD_ID=...
CLIENT_SECRET = os.getenv("ICD_SECRET")  # export ICD_SECRET=...
if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Error: Please set ICD_ID / ICD_SECRET first")

TOKEN_EP = "https://icdaccessmanagement.who.int/connect/token"
SCOPE    = "icdapi_access"
BASE = "https://id.who.int/icd/release/11/2024-01/mms"
HEADERS  = {"Accept": "application/json",
            "Accept-Language": "en",     # English
            "API-Version": "v2"}         # Recommended to add v2

# ------------------------------------------------------------------
# 2. Get access_token
def get_token() -> str:
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": SCOPE,
        "grant_type": "client_credentials"
    }
    r = requests.post(TOKEN_EP, data=payload, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

# ------------------------------------------------------------------
# 3. Get Chapter 6 and all its child entities recursively
def get_all_chapter6_entities(token: str) -> List[str]:
    # First, get Chapter 6 data
    chapter6_id = "334423054"  # Known Chapter 6 ID
    url = f"{BASE}/{chapter6_id}"
    hdrs = {"Authorization": f"Bearer {token}", **HEADERS}
    params = {"lang": "en"}
    
    print(f"   Getting Chapter 6 data...")
    r = requests.get(url, headers=hdrs, params=params, timeout=60)
    r.raise_for_status()
    chapter6_data = r.json()
    
    print(f"   Chapter 6 title: {chapter6_data.get('title', {}).get('@value', 'Unknown')}")
    
    # Extract all child entities recursively
    all_entity_ids = set()
    processed_entities = set()
    
    def extract_children_recursively(entity_id: str, depth: int = 0):
        if entity_id in processed_entities:
            return
        
        processed_entities.add(entity_id)
        indent = "  " * depth
        
        try:
            print(f"{indent}Processing entity: {entity_id}")
            entity_data = get_entity(token, entity_id)
            
            # Add current entity to the list
            all_entity_ids.add(entity_id)
            
            # Save raw JSON
            raw_dir = pathlib.Path("icd11_ch6_data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{entity_id}.json").write_text(
                json.dumps(entity_data, ensure_ascii=False, indent=2))
            
            # Extract child entities
            if "child" in entity_data and isinstance(entity_data["child"], list):
                children = entity_data["child"]
                print(f"{indent}Found {len(children)} children")
                
                for child in children:
                    if isinstance(child, str):
                        # Extract the last part of the URL as entity ID
                        if child.startswith("http"):
                            parsed = urlparse(child)
                            child_id = parsed.path.split('/')[-1]
                            # Skip "other" and "unspecified" entries
                            if child_id not in ["other", "unspecified"]:
                                extract_children_recursively(child_id, depth + 1)
                        else:
                            extract_children_recursively(child, depth + 1)
                            
        except Exception as e:
            print(f"{indent}Error processing {entity_id}: {e}")
    
    # Start recursive extraction from Chapter 6
    extract_children_recursively(chapter6_id)
    
    print(f"   Found {len(all_entity_ids)} total entities")
    return list(all_entity_ids)

# ------------------------------------------------------------------
# 4. Download detailed JSON for single entry
def get_entity(token: str, code: str) -> Dict:
    url  = f"{BASE}/{code}"
    hdrs = {"Authorization": f"Bearer {token}", **HEADERS}
    r = requests.get(url, headers=hdrs, params={"lang": "en"}, timeout=30)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------------
def main():
    out_dir = pathlib.Path("icd11_ch6_data")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("-> Requesting OAuth2 token ...")
    token = get_token()

    print("-> Fetching all Chapter 6 entities recursively ...")
    entity_ids = get_all_chapter6_entities(token)
    print(f"   Found {len(entity_ids)} total mental health disorders and subcategories")

    csv_rows = []
    for idx, code in enumerate(entity_ids, 1):
        # Progress bar
        print(f"[{idx:3}/{len(entity_ids)}] Processing {code} for CSV", end="\r")
        
        try:
            # Read the JSON file we already saved
            json_file = raw_dir / f"{code}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # If file doesn't exist, fetch it again
                data = get_entity(token, code)
                (raw_dir / f"{code}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2))

            # Extract key fields for CSV - using correct JSON structure
            title = ""
            definition = ""
            inclusions = ""
            exclusions = ""
            code_range = ""
            class_kind = ""
            
            # Extract title
            if "title" in data and isinstance(data["title"], dict):
                title = data["title"].get("@value", "")
            
            # Extract definition
            if "definition" in data and isinstance(data["definition"], dict):
                definition = data["definition"].get("@value", "")
            
            # Extract code range
            if "codeRange" in data:
                code_range = data["codeRange"]
            
            # Extract class kind
            if "classKind" in data:
                class_kind = data["classKind"]
            
            # Extract inclusions
            if "inclusion" in data and isinstance(data["inclusion"], list):
                inclusion_texts = []
                for inc in data["inclusion"]:
                    if isinstance(inc, dict) and "label" in inc:
                        label = inc["label"]
                        if isinstance(label, dict) and "@value" in label:
                            inclusion_texts.append(label["@value"])
                inclusions = "|".join(inclusion_texts)
            
            # Extract exclusions
            if "exclusion" in data and isinstance(data["exclusion"], list):
                exclusion_texts = []
                for exc in data["exclusion"]:
                    if isinstance(exc, dict) and "label" in exc:
                        label = exc["label"]
                        if isinstance(label, dict) and "@value" in label:
                            exclusion_texts.append(label["@value"])
                exclusions = "|".join(exclusion_texts)
            
            row = {
                "code"      : code,
                "title"     : title,
                "definition": definition,
                "code_range": code_range,
                "class_kind": class_kind,
                "inclusions": inclusions,
                "exclusions": exclusions,
            }
            csv_rows.append(row)
            
        except Exception as e:
            print(f"\nWarning: {code} failed: {e}")
            continue

    # Write CSV
    if csv_rows:
        csv_file = out_dir / "icd11_ch6_entities.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nDone!  Saved {len(csv_rows)} mental health disorders and subcategories to {csv_file}")
    else:
        print(f"\nNo data to save")

if __name__ == "__main__":
    main()
