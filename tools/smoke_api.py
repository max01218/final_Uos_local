# tools/smoke_api.py
import requests, json
URL="http://localhost:8000/api/v2/empathetic_professional"; SID="sanity-1"
def hit(q,tone="balanced"):
    r=requests.post(URL,json={"question":q,"type":tone,"session_id":SID},timeout=30); r.raise_for_status()
    data=r.json(); print("\nQ>",q); print("A>\n",data.get("answer")); 
    print("Validator>", json.dumps(data.get("meta",{}).get("validator",{}),ensure_ascii=False))
hit("hi"); hit("I feel anxious about sleep","caring"); hit("6","caring")
