from collections import Counter
from typing import List

TOPIC_KWS = {
    "sleep":  {"insomnia","cbt-i","stimulus control","sleep hygiene","sleep restriction","20-minute rule"},
    "panic":  {"panic attack","hyperventilation","heart racing","shortness of breath"},
    "anxiety":{"anxiety","worry","rumination","catastroph","restless"},
    "trauma": {"flashback","grounding","5-4-3-2-1","triggered"},
    "grief":  {"bereavement","grief","loss"},
    "depress":{"anhedonia","low mood","hopeless","fatigue"},
}
TECHNIQUE_BY_TOPIC = {
    "sleep":"stimulus_control","panic":"breathing","trauma":"grounding",
    "grief":"grounding","anxiety":"grounding","depress":"pmr",
}
def infer_topics(docs)->List[str]:
    bag=Counter()
    for d in (docs or []):
        meta=getattr(d,"metadata",{}) or {}
        text=(" ".join([meta.get("title",""),meta.get("code",""),meta.get("tags",""),
                        getattr(d,"page_content","") or ""])).lower()
        for topic,kws in TOPIC_KWS.items():
            for kw in kws:
                if kw in text: bag[topic]+=1
    return [k for k,_ in bag.most_common()]
def choose_technique(topics:List[str], last_technique:str|None, continue_flag:bool)->str:
    if continue_flag and last_technique: return last_technique
    if topics: return TECHNIQUE_BY_TOPIC.get(topics[0],"grounding")
    return "grounding"
def step_map_for(technique:str)->str:
    if technique=="grounding":
        return ("- grounding (5-4-3-2-1):\n"
                "  1: Name 5 things you can see — 60 seconds.\n"
                "  2: Touch 4 different textures — 60 seconds.\n"
                "  3: Listen for 3 distinct sounds — 30 seconds.\n"
                "  4: Notice 2 smells — 30 seconds.\n"
                "  5: Name 1 thing you can taste — 10 seconds.\n")
    if technique=="breathing":
        return ("- breathing:\n"
                "  1: Inhale 4, hold 2, exhale 6 — 4 cycles.\n"
                "  2: Repeat the same pattern — 4 cycles.\n"
                "  3: Place a hand on belly; feel rise/fall for 4 breaths.\n")
    if technique=="pmr":
        return ("- pmr:\n"
                "  1: Tense hands 5s, release 10s — 2 times.\n"
                "  2: Tense shoulders 5s, release 10s — 2 times.\n"
                "  3: Tense jaw 5s, release 10s — 2 times.\n")
    if technique=="stimulus_control":
        return ("- stimulus_control (sleep):\n"
                "  1: If awake >20 min, leave bed; do a quiet task 10–15 min.\n"
                "  2: Return to bed when sleepy; repeat once if needed.\n")
    return ""
