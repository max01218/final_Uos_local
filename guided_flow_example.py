# guided_flow_example.py
"""
Example demonstrating the guided flow system for mh_support route.

This system implements micro-step interventions with the following features:
- One micro-step per turn (S line) with explicit timing/repetitions
- One question per turn (Q line) 
- Progress tracking through step_index and user feedback
- Automatic completion after 3-5 steps with wrap-up
- State management through ConversationStore flags
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.chat_service import ChatService
from app.services.memory_service import ConversationStore
from app.schemas.chat import RAGRequest

async def demonstrate_guided_flow():
    """Demonstrate the complete guided flow functionality"""
    
    print("=== Guided Flow Demonstration ===")
    print("This shows how mh_support route uses micro-step interventions\n")
    
    # Initialize the chat service
    cs = ConversationStore()
    chat_service = ChatService(conversation_store=cs)
    
    # Conversation flow
    conversation = [
        "I'm feeling really anxious about my job interview tomorrow",
        "7",  # Rating response
        "done",  # Continue signal
        "I think I can do that",  # Positive response
        "6",  # Improved rating
    ]
    
    session_id = "demo_session_001"
    
    for i, user_input in enumerate(conversation):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {user_input}")
        
        # Create request
        request = RAGRequest(
            question=user_input,
            session_id=session_id,
            type="balanced"
        )
        
        # Get response
        try:
            response, metadata = await chat_service.handle_chat(request)
            print(f"Assistant: {response}")
            print(f"Metadata: {metadata}")
            
            # Show flow state if active
            if chat_service.orch.flow:
                state = chat_service.orch.flow.load_state(session_id)
                if state.active:
                    print(f"Flow State - Technique: {state.technique}, Step: {state.step_index}, Rating: {state.last_rating}")
            
        except Exception as e:
            print(f"Error: {e}")
            break
        
        # Small delay for readability
        await asyncio.sleep(0.5)
    
    print("\n=== Demo Complete ===")
    print("Key features demonstrated:")
    print("- Automatic technique selection and planning")
    print("- Micro-step progression with E:S:Q: format")
    print("- User feedback processing (ratings, continue signals)")
    print("- State management across conversation turns")
    print("- Judge/repair for response quality")

if __name__ == "__main__":
    asyncio.run(demonstrate_guided_flow())
