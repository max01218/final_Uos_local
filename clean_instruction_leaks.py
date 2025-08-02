import json
import re

def clean_instruction_leaks():
    """Clean internal instruction leaks from interactions.json"""
    
    # Load interactions
    try:
        with open('interactions.json', 'r', encoding='utf-8') as f:
            interactions = json.load(f)
    except FileNotFoundError:
        print("interactions.json not found")
        return
    except json.JSONDecodeError:
        print("Error reading interactions.json")
        return
    
    print(f"Loaded {len(interactions)} interactions")
    
    # Patterns to remove
    internal_instruction_patterns = [
        r'INSTRUCTIONS?:.*?(?=\n|$)',  # "INSTRUCTIONS: - Start with empathy..."
        r'INSTRUCTATIONS?:.*?(?=\n|$)',  # Typo version
        r'RESPONSE TEMPLATE:.*?(?=\n|$)',  # "RESPONSE TEMPLATE:"
        r'GUIDELINES:.*?(?=\n|$)',  # "GUIDELINES:"
        r'Start with empathy.*?(?=\n|$)',  # "Start with empathy (1 sentence)"
        r'Cite ICD-11.*?(?=\n|$)',  # "Cite ICD-11 context if relevant"
        r'Ask 1 gentle.*?(?=\n|$)',  # "Ask 1 gentle follow-up question"
        r'Keep response.*?(?=\n|$)',  # "Keep response to 2-4 sentences"
        r'Avoid generic.*?(?=\n|$)',  # "Avoid generic lifestyle advice"
        r'Response Structure.*?(?=\n|$)',  # "Response Structure Guidelines:"
        r'Formatting Guidelines.*?(?=\n|$)',  # "Formatting Guidelines:"
        r'Content Depth.*?(?=\n|$)',  # "Content Depth Guidelines:"
        r'Professional Resource.*?(?=\n|$)',  # "Professional Resource Guidelines:"
        r'Balance Guidelines.*?(?=\n|$)',  # "Balance Guidelines:"
        r'Question Type.*?(?=\n|$)',  # "Question Type Adaptations:"
        r'Quality Standards.*?(?=\n|$)',  # "Quality Standards:"
        r'Personalization Elements.*?(?=\n|$)',  # "Personalization Elements:"
        r'Response Template.*?(?=\n|$)',  # "Response Template Structure:"
        r'1\. Empathy.*?(?=\n|$)',  # "1. Empathy Opening"
        r'2\. Problem.*?(?=\n|$)',  # "2. Problem Acknowledgment"
        r'3\. Structured.*?(?=\n|$)',  # "3. Structured Advice"
        r'4\. Professional.*?(?=\n|$)',  # "4. Professional Resources"
        r'5\. Encouragement.*?(?=\n|$)',  # "5. Encouragement Closing"
        r'Crisis Questions.*?(?=\n|$)',  # "Crisis Questions:"
        r'How-to Questions.*?(?=\n|$)',  # "How-to Questions:"
        r'Symptom Questions.*?(?=\n|$)',  # "Symptom Questions:"
        r'General Support.*?(?=\n|$)',  # "General Support:"
        r'Each piece of advice.*?(?=\n|$)',  # "Each piece of advice should be"
        r'Include the reasoning.*?(?=\n|$)',  # "Include the reasoning behind"
        r'Provide multiple options.*?(?=\n|$)',  # "Provide multiple options"
        r'Ensure advice is.*?(?=\n|$)',  # "Ensure advice is evidence-based"
        r'Include appropriate.*?(?=\n|$)',  # "Include appropriate disclaimers"
        r'Reference user.*?(?=\n|$)',  # "Reference user's specific situation"
        r'Adapt advice based.*?(?=\n|$)',  # "Adapt advice based on emotional state"
        r'Consider cultural.*?(?=\n|$)',  # "Consider cultural and contextual"
        r'Provide age-appropriate.*?(?=\n|$)',  # "Provide age-appropriate"
    ]
    
    cleaned_count = 0
    
    for interaction in interactions:
        if 'answer' in interaction:
            original_answer = interaction['answer']
            cleaned_answer = original_answer
            
            # Apply all patterns
            for pattern in internal_instruction_patterns:
                cleaned_answer = re.sub(pattern, '', cleaned_answer, flags=re.IGNORECASE | re.MULTILINE)
            
            # Clean up extra whitespace
            cleaned_answer = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_answer)
            cleaned_answer = cleaned_answer.strip()
            
            if cleaned_answer != original_answer:
                interaction['answer'] = cleaned_answer
                cleaned_count += 1
                print(f"Cleaned interaction {interactions.index(interaction) + 1}")
    
    # Save cleaned interactions
    with open('interactions.json', 'w', encoding='utf-8') as f:
        json.dump(interactions, f, indent=2, ensure_ascii=False)
    
    print(f"\nCleaned {cleaned_count} interactions")
    print("interactions.json updated")

if __name__ == "__main__":
    clean_instruction_leaks() 