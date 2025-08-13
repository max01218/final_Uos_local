import re
from typing import List
from app.schemas.chat import Message


def analyze_conversation_context(question: str, history: List[Message]) -> dict:
    analysis = {
        "information_level": "basic",
        "response_strategy": "ask_clarification",
        "already_discussed": [],
        "key_topics": [],
        "emotional_state": "neutral",
    }
    question_lower = question.lower()

    detailed_patterns = [
        r"because\s+\w+",
        r"since\s+\w+",
        r"for\s+\d+",
        r"every\s+\w+",
        r"always\s+\w+",
        r"constantly\s+\w+",
        r"trouble\s+\w+",
        r"can't\s+\w+",
        r"won't\s+\w+",
        r"almost\s+\w+",
    ]
    detailed_count = 0
    for pattern in detailed_patterns:
        if re.search(pattern, question_lower):
            detailed_count += 1

    comprehensive_patterns = [
        r"and\s+\w+.*and\s+\w+",
        r"not\s+only.*but\s+also",
        r"both\s+\w+.*and\s+\w+",
        r"either\s+\w+.*or\s+\w+",
    ]
    comprehensive_count = 0
    for pattern in comprehensive_patterns:
        if re.search(pattern, question_lower):
            comprehensive_count += 1

    if comprehensive_count >= 1 or detailed_count >= 3:
        analysis["information_level"] = "comprehensive"
        analysis["response_strategy"] = "provide_advice"
    elif detailed_count >= 1:
        analysis["information_level"] = "detailed"
        analysis["response_strategy"] = "provide_advice"
    else:
        analysis["information_level"] = "basic"
        analysis["response_strategy"] = "ask_clarification"

    how_to_patterns = [
        r'how to\s+\w+',
        r'how do i\s+\w+',
        r'what should i do\s+\w+',
        r'what can i do\s+\w+',
        r'how can i\s+\w+',
        r'what steps\s+\w+',
        r'what techniques\s+\w+',
        r'can you show me\s+\w+',
        r'can you tell me\s+\w+',
    ]
    mental_health_conditions = [
        'anxiety', 'anxious', 'depression', 'depressed', 'stress', 'stressed',
        'worry', 'worried', 'panic', 'fear', 'afraid', 'nervous',
        'thoughts', 'thinking', 'behavior', 'mood', 'overwhelmed',
        'sad', 'down', 'upset', 'tense', 'release', 'relieve', 'reduce',
        'cope', 'manage', 'deal', 'handle', 'overcome',
    ]
    is_how_to_question = any(re.search(pattern, question_lower) for pattern in how_to_patterns)
    has_mental_health_content = any(condition in question_lower for condition in mental_health_conditions)
    if is_how_to_question and has_mental_health_content:
        analysis["response_strategy"] = "provide_advice"
        if detailed_count >= 1:
            analysis["response_strategy"] = "give_specific_help"

    simple_greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', "how's it going", 'nice to meet you', 'pleasure to meet you',
    ]
    if any(greeting in question_lower for greeting in simple_greetings) and len(question_lower.split()) <= 3:
        analysis["response_strategy"] = "ask_clarification"
        analysis["information_level"] = "basic"

    emotional_keywords = {
        "anxiety": ["anxious", "anxiety", "worried", "worry", "fear", "afraid", "scared"],
        "depression": ["sad", "depressed", "hopeless", "worthless", "empty", "numb"],
        "stress": ["stressed", "overwhelmed", "pressure", "tension", "burnout"],
        "anger": ["angry", "frustrated", "irritated", "mad", "upset"],
        "fear": ["terrified", "panic", "panic attack", "frightened"],
    }
    for emotion, keywords in emotional_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            analysis["emotional_state"] = emotion
            break

    topic_keywords = {
        "work": ["work", "job", "career", "office", "boss", "colleague", "deadline"],
        "sleep": ["sleep", "insomnia", "tired", "exhausted", "rest", "bed"],
        "relationships": ["relationship", "partner", "family", "friend", "marriage"],
        "health": ["health", "medical", "doctor", "symptoms", "pain", "illness"],
        "finances": ["money", "financial", "bills", "debt", "expenses", "salary"],
    }
    for topic, keywords in topic_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            analysis["key_topics"].append(topic)

    if history:
        recent_history = history[-3:]
        for msg in recent_history:
            if msg.role == "user":
                for topic in analysis["key_topics"]:
                    if topic in msg.content.lower():
                        analysis["already_discussed"].append(topic)

    if analysis["already_discussed"] and analysis["information_level"] in ["detailed", "comprehensive"]:
        analysis["response_strategy"] = "give_specific_help"

    return analysis



