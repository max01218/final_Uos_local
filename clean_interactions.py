import json
import re

def clean_interactions():
    print("Cleaning interactions.json...")
    
    try:
        # Load interactions
        with open('interactions.json', 'r', encoding='utf-8') as f:
            interactions = json.load(f)
        
        print(f"Loaded {len(interactions)} interactions")
        
        # Clean each interaction
        cleaned_interactions = []
        for interaction in interactions:
            answer = interaction.get('answer', '')
            
            # Remove format tags
            answer = re.sub(r'<\|assistant\|\|?\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'<\|.*?\|>', '', answer, flags=re.IGNORECASE)
            
            # Remove debug patterns
            answer = re.sub(r'User:.*?(?=\n|$)', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'Assistant asked:.*?(?=\n|$)', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'Assistant:.*?(?=\n|$)', '', answer, flags=re.IGNORECASE)
            
            # Remove repetitive phrases
            repetitive_patterns = [
                r'Please consider taking breaks throughout the day\. It doesn\'t have to be an hour.*?',
                r'It doesn\'t have to be a big deal.*?',
                r'just a little bit of extra time will go a long way.*?',
                r'Remember to take breaks and rest when you need to.*?',
                r'It\'s important to prioritize your safety and well-being.*?',
            ]
            
            for pattern in repetitive_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)
            
            # Clean up whitespace
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            # Skip if answer is too short or empty
            if len(answer) < 10:
                continue
                
            # Update the interaction
            interaction['answer'] = answer
            cleaned_interactions.append(interaction)
        
        print(f"Cleaned to {len(cleaned_interactions)} interactions")
        
        # Save cleaned interactions
        with open('interactions.json', 'w', encoding='utf-8') as f:
            json.dump(cleaned_interactions, f, indent=2, ensure_ascii=False)
        
        print("✅ Interactions cleaned and saved!")
        
    except Exception as e:
        print(f"❌ Error cleaning interactions: {e}")

if __name__ == "__main__":
    clean_interactions() 