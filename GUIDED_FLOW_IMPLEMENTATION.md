# Guided Flow Implementation for Mental Health Support

## Overview

This implementation provides a guided micro-step intervention system for the `mh_support` route, delivering structured mental health support through conversation turns that follow a strict E:S:Q format.

## Key Features

### 1. Micro-Step Interventions
- **One micro-step per turn**: Each turn provides exactly one actionable step (S line)
- **Explicit timing/repetitions**: All steps include specific durations or counts
- **Progressive difficulty**: Steps build on each other within a technique

### 2. Question Management
- **One question per turn**: Each response ends with exactly one question (Q line)
- **Question type control**: Uses `rating_0_10` or `yes_no` patterns
- **User feedback processing**: Automatically detects ratings, continue signals, and difficulty indicators

### 3. State Management
- **Technique tracking**: Maintains current intervention technique
- **Step progression**: Tracks progress through 3-5 step sequences
- **Session persistence**: State survives across conversation turns
- **Automatic completion**: Flows complete with wrap-up after maximum steps

## Implementation Architecture

### Core Components

#### 1. GuidedFlowService (`app/orchestration/flow_service.py`)
- **FlowState**: Dataclass managing flow state flags
- **Planning**: Initial technique selection and step planning
- **Turn generation**: Produces guided responses using prompt compilation
- **State progression**: Handles user feedback and step advancement
- **Completion detection**: Manages flow wrap-up and reset

#### 2. PromptCompiler Extensions (`app/orchestration/prompt_compiler.py`)
- **Flow prompts**: Specialized prompt compilation for guided flows
- **Template management**: Reads and formats flow-specific prompt templates
- **JSON extraction**: Parses technique and step data from LLM responses

#### 3. Orchestrator Integration (`app/orchestration/orchestrator.py`)
- **Route detection**: Identifies mh_support requests
- **Flow handling**: Delegates to GuidedFlowService for flow-enabled routes
- **Judge/repair**: Maintains quality control for flow responses

#### 4. ChatService Updates (`app/services/chat_service.py`)
- **Feedback processing**: Handles user input for flow progression
- **State advancement**: Updates flow state based on user responses
- **Session management**: Maintains flow state across conversation turns

### Data Structures

#### Flow State Flags (stored in ConversationStore)
```python
flow_active: bool           # Whether guided flow is currently active
technique: str             # Current technique (e.g., "breathing_box")
step_index: int           # Current step position (0-based)
last_rating: Optional[int] # User's last 0-10 rating
last_question_type: str   # Expected response type
plan_json: str           # Original technique plan
total_steps: int         # Total steps in current plan
```

#### Technique Database (`app/prompts/data/mh_support_techniques.json`)
```json
{
  "technique_name": {
    "name": "Display Name",
    "description": "Brief description",
    "steps": [
      {
        "action": "Specific action to take",
        "duration": "Time requirement",
        "detail": "Additional guidance"
      }
    ]
  }
}
```

### Prompt Templates

#### 1. Planning Prompt (`app/prompts/flows/mh_support_plan_prompt.md`)
- **Purpose**: Initial technique selection and step planning
- **Output**: JSON with technique and 3-5 steps
- **Input**: User's anxiety description and conversation history

#### 2. Turn Prompt (`app/prompts/flows/mh_support_turn_prompt.md`)
- **Purpose**: Generate guided turn responses
- **Output**: Strict E:S:Q: format
- **Input**: Current step, technique, plan, question type

#### 3. Adjustment Prompt (`app/prompts/flows/mh_support_adjust_prompt.md`)
- **Purpose**: Handle user difficulty or stuck situations
- **Output**: Modified E:S:Q: with reduced burden
- **Input**: Problem description and current technique

#### 4. Wrap-up Prompt (`app/prompts/flows/mh_support_wrap_up_prompt.md`)
- **Purpose**: Session completion and maintenance suggestions
- **Output**: Final E:S:Q: with practice recommendation
- **Input**: Completed technique and session history

## User Interaction Flow

### 1. Initial Request
- User expresses anxiety or mental health concern
- Router classifies as `mh_support` route
- GuidedFlowService creates initial plan using planning prompt
- First guided response generated with step 0

### 2. Micro-Step Progression
- User provides feedback (rating, "done", "continue", etc.)
- ChatService processes feedback before next turn
- FlowService advances or adjusts step_index based on feedback
- Next guided response generated for current step

### 3. Feedback Processing
- **Ratings 0-3**: May step back or stay on current step
- **Ratings 4-10**: Advance to next step
- **"done"/"continue"**: Advance to next step
- **Difficulty indicators**: Switch to adjustment mode

### 4. Completion
- After 3-5 steps, flow enters wrap-up mode
- Wrap-up provides summary and maintenance suggestions
- Flow state resets to inactive

## Available Techniques

1. **grounding_visual**: Focus on immediate environment, visual anchoring
2. **breathing_box**: Structured breathing patterns with counts
3. **PMR_hand**: Progressive muscle relaxation for hands/arms
4. **cognitive_reframe**: Simple thought challenging
5. **mindful_pause**: Brief mindfulness moments

Each technique includes 3-5 progressive steps with explicit timing and clear instructions.

## Quality Control

### Judge/Repair Integration
- All flow responses pass through existing judge/repair system
- Validates E:S:Q: format compliance
- Ensures single question per turn
- Maintains word count limits (25 words per line)

### Constraint Enforcement
- **real_voice**: Natural, conversational language
- **one_question**: Exactly one question per response
- **no_cliches**: Avoids therapeutic jargon
- **no_emoji**: Professional text only
- **explicit_timing**: All steps include timing/repetitions

## Usage Example

```python
# Initialize with conversation store
cs = ConversationStore()
chat_service = ChatService(conversation_store=cs)

# User starts with anxiety
request = RAGRequest(
    question="I'm feeling anxious about my presentation",
    session_id="user_123"
)

# First response: Planning + initial step
response, meta = await chat_service.handle_chat(request)
# Output: E:... S: Take slow breath in for 4 seconds Q: Rate anxiety 0-10?

# User provides rating
request.question = "I'd say it's a 7"
response, meta = await chat_service.handle_chat(request)
# Output: E:... S: Hold breath for 4 seconds Q: How does that feel?

# Continue through 3-5 steps until completion
```

## Files Modified/Created

### New Files
- `app/orchestration/flow_service.py` - Core flow management
- `app/prompts/flows/mh_support_plan_prompt.md` - Planning template
- `app/prompts/flows/mh_support_turn_prompt.md` - Turn guidance template
- `app/prompts/flows/mh_support_adjust_prompt.md` - Adjustment template
- `app/prompts/flows/mh_support_wrap_up_prompt.md` - Wrap-up template
- `app/prompts/data/mh_support_techniques.json` - Technique database

### Modified Files
- `app/orchestration/orchestrator.py` - Added flow service integration
- `app/orchestration/prompt_compiler.py` - Added flow prompt compilation
- `app/services/chat_service.py` - Added feedback processing
- `app/services/memory_service.py` - Already had flag storage capability

## Benefits

1. **Structured Support**: Provides systematic, step-by-step interventions
2. **User Control**: Allows users to control pacing through feedback
3. **Quality Assurance**: Maintains consistent format and therapeutic boundaries
4. **Scalable**: Easy to add new techniques and modify existing ones
5. **Stateful**: Remembers progress across conversation turns
6. **Flexible**: Adapts to user feedback and difficulty levels

This implementation successfully delivers the requested "micro-step mental health support" with strict format control, user feedback processing, and automatic progression management.
