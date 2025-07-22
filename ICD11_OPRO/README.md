# ICD-11 OPRO Prompt Optimization System

This is a standalone OPRO (Optimization by PROmpting) system designed specifically for optimizing prompts for the ICD-11 mental health RAG system.

## 📁 Project Structure

```
ICD11_OPRO/
├── README.md                      # Project documentation
├── run_opro.py                   # Main execution script
├── evaluate_prompt.py            # Prompt evaluation module
├── requirements.txt              # Dependencies list
├── config.json                   # Configuration file
├── opro/
│   ├── __init__.py
│   ├── optimize_icd11_prompt.py  # Core OPRO optimization logic
│   └── seed_prompts/             # Seed prompts
│       ├── seed_empathetic.txt   # Empathetic version
│       ├── seed_professional.txt # Professional version
│       └── seed_crisis.txt       # Crisis intervention version
├── prompts/                      # Output directory
│   ├── optimized_prompt.txt      # Optimized prompt
│   └── optimization_history.json # Optimization history
└── tests/                        # Test cases
    └── test_cases.json           # Standard test set
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ICD11_OPRO
pip install -r requirements.txt
```

### 2. Run Optimization
```bash
python run_opro.py
```

### 3. Select Operation Mode
- **Mode 1**: Run OPRO optimization - Automatically optimize prompts
- **Mode 2**: Evaluate existing prompts - Test prompt performance  
- **Mode 3**: View optimization history - Monitor optimization progress

## 📊 Evaluation Metrics

The system evaluates prompt quality using four dimensions:

- **Relevance (25%)**: Whether the response directly answers user questions
- **Empathy (30%)**: Whether it demonstrates emotional support and understanding
- **Accuracy (25%)**: Whether medical information is accurate and reliable
- **Safety (20%)**: Whether it appropriately handles risk situations

## 🔧 Configuration

Edit `config.json` to adjust optimization parameters:

- `max_iterations`: Maximum optimization iterations
- `improvement_threshold`: Minimum improvement threshold
- `evaluation_weights`: Weights for each evaluation dimension
- `safety_requirements`: Safety requirement settings

## 🔄 Main System Integration

After optimization is complete, integrate the generated `prompts/optimized_prompt.txt` file into your main RAG system:

```python
# Load the optimized prompt in your main system
with open('ICD11_OPRO/prompts/optimized_prompt.txt', 'r') as f:
    optimized_prompt = f.read()
```

## 📋 Notes

1. Ensure seed prompts contain appropriate mental health professional standards
2. Regularly check optimization history to monitor system performance
3. For production environments, recommend validating optimization results in test environment first
4. Keep backups of original prompts for rollback if necessary 