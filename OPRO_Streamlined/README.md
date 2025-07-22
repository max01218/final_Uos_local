# Streamlined OPRO System

A clean, streamlined version of the OPRO (Optimization by PROmpting) system for mental health prompt optimization. This version removes all Chinese characters, emojis, and redundant files for better maintainability.

## Features

- **Clean Codebase**: No Chinese characters or emojis in code
- **Streamlined Architecture**: Removed redundant and duplicate files
- **Automatic Optimization**: Scheduled prompt optimization based on user feedback
- **Local LLM Support**: Uses Llama 3 for optimization without requiring external APIs
- **Integrated System**: Works with FastAPI servers and feedback collection

## Directory Structure

```
OPRO_Streamlined/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_opro.py                 # Main execution script
├── config/
│   └── config.json             # System configuration
├── core/
│   ├── opro_optimizer.py       # Core OPRO optimization logic
│   ├── scheduler.py            # Automatic scheduling system
│   └── feedback_converter.py   # Convert feedback to test cases
├── prompts/
│   ├── seeds/                  # Seed prompts for optimization
│   ├── optimized_prompt.txt    # Latest optimized prompt
│   └── backups/                # Backup prompts
├── tests/
│   └── feedback_testcases_*.json # Generated test cases
└── logs/
    ├── opro_scheduler.log      # Scheduler logs
    └── opro_scheduler_state.json # Scheduler state
```

## Quick Start

### 1. Install Dependencies

```bash
cd OPRO_Streamlined
pip install -r requirements.txt
```

### 2. Check System Status

```bash
python run_opro.py --mode info
```

### 3. Run Optimization

```bash
python run_opro.py --mode optimize
```

### 4. Start Scheduler (Automatic Mode)

```bash
python core/scheduler.py
```

## Usage

### Command Line Interface

```bash
# Display system information
python run_opro.py --mode info

# Run single optimization
python run_opro.py --mode optimize

# Evaluate a specific prompt
python run_opro.py --mode evaluate --prompt-file prompts/optimized_prompt.txt
```

### Scheduler Operations

```bash
# Run scheduler once (check and optimize if needed)
python core/scheduler.py --run-once

# Run continuous scheduler (default: check every 24 hours)
python core/scheduler.py

# Custom configuration
python core/scheduler.py --min-interactions 5 --check-interval 12
```

### Manual Feedback Conversion

```bash
# Convert interactions.json to test cases
python core/feedback_converter.py --input interactions.json
```

## Configuration

Edit `config/config.json` to customize:

```json
{
    "opro_settings": {
        "max_iterations": 5,
        "improvement_threshold": 0.05,
        "early_stopping_patience": 2,
        "temperature": 0.7,
        "max_tokens": 512
    },
    "evaluation": {
        "weights": {
            "relevance": 0.25,
            "empathy": 0.30,
            "accuracy": 0.25,
            "safety": 0.20
        },
        "passing_threshold": 7.0
    }
}
```

## Integration with FastAPI

The optimized prompts are automatically saved to `prompts/optimized_prompt.txt` and can be loaded by your FastAPI server:

```python
# In your FastAPI server
def load_opro_prompt():
    try:
        with open('OPRO_Streamlined/prompts/optimized_prompt.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return fallback_prompt
```

## Automatic Workflow

1. **Feedback Collection**: User interactions and feedback are saved to `interactions.json`
2. **Conversion**: Scheduler converts feedback to test cases using `feedback_converter.py`
3. **Optimization**: OPRO runs optimization using converted test cases
4. **Deployment**: New optimized prompt is saved and can be used immediately

## Key Improvements

- **Removed Chinese Content**: All Chinese characters and comments replaced with English
- **No Emojis**: Clean, professional code without emoji characters
- **Simplified Structure**: Removed duplicate and redundant files
- **Better Organization**: Clear separation of core logic, configuration, and outputs
- **Enhanced Logging**: Comprehensive logging system for debugging
- **Modular Design**: Each component is independent and can be used separately

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for Llama 3)
- At least 8GB RAM
- 10GB disk space for model storage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_tokens` in configuration
2. **Model Loading Error**: Check internet connection for first-time model download
3. **No Test Cases Generated**: Ensure `interactions.json` exists and contains valid feedback

### Log Files

- Scheduler logs: `logs/opro_scheduler.log`
- Optimization history: `prompts/optimization_history.json`
- System state: `logs/opro_scheduler_state.json`

## License

This is a streamlined version of the original OPRO system for educational and research purposes. 