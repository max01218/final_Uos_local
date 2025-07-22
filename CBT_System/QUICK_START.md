# CBT System Quick Start Guide

## What is this?

This is a complete Cognitive Behavioral Therapy (CBT) knowledge base system that can enhance your mental health applications with evidence-based therapeutic techniques.

## Step 1: Install Dependencies

```bash
# Install required Python packages
pip install requests beautifulsoup4 sentence-transformers faiss-cpu numpy
```

## Step 2: Set Up the System

```bash
# Navigate to CBT_System folder
cd CBT_System

# Run automated setup
python setup.py
```

This will automatically:
- Check all dependencies
- Collect CBT data from public sources (NHS, NIMH, etc.)
- Process and clean the data
- Create a searchable vector database
- Test the integration

**Note**: Setup may take 15-30 minutes depending on your internet speed and computer performance.

## Step 3: Test the System

```bash
# Test all components
python test_system.py
```

This will verify that everything is working correctly.

## Step 4: Basic Usage

### Simple Enhancement Example

```python
from integration import CBTIntegration

# Initialize CBT system
cbt = CBTIntegration()

# Check if system is ready
status = cbt.get_cbt_status()
print(f"CBT Available: {status['available']}")

# Enhance a response with CBT techniques
user_question = "I feel anxious all the time, what can I do?"
base_response = "I understand you're feeling overwhelmed."

enhanced_response = cbt.enhance_response_with_cbt(
    user_query=user_question,
    context="",
    base_response=base_response
)

print(enhanced_response)
```

### Direct CBT Recommendations

```python
# Get specific CBT technique recommendations
recommendations = cbt.cbt_kb.get_cbt_recommendation(
    "How can I challenge negative thoughts?"
)

# Format into a professional response
response = cbt.cbt_kb.format_cbt_response(
    recommendations, 
    "How can I challenge negative thoughts?"
)

print(response)
```

## What You Get

After setup, your system will have:

### 6 CBT Technique Categories
1. **Cognitive Restructuring** - Thought challenging, identifying distortions
2. **Behavioral Activation** - Activity scheduling, behavioral experiments
3. **Exposure Therapy** - Systematic desensitization, gradual exposure
4. **Problem-Solving** - Structured approaches to challenges
5. **Relaxation Techniques** - Breathing, progressive muscle relaxation
6. **Psychoeducation** - Mental health education and awareness

### Smart Query Analysis
The system automatically detects when CBT techniques would be helpful based on:
- User question patterns ("how can I", "help me cope")
- Mental health indicators (anxiety, depression, stress)
- Specific technique requests

### Professional Safety
- Includes appropriate disclaimers
- Encourages professional consultation
- Maintains therapeutic boundaries
- Uses only evidence-based techniques

## Integration with Your System

### FastAPI Example

```python
from fastapi import FastAPI
from integration import CBTIntegration

app = FastAPI()
cbt = CBTIntegration()

@app.post("/chat")
async def chat_endpoint(request: dict):
    user_query = request.get("question", "")
    base_response = "I understand your concern."
    
    # Enhance with CBT if relevant
    enhanced_response = cbt.enhance_response_with_cbt(
        user_query=user_query,
        context="",
        base_response=base_response
    )
    
    return {"answer": enhanced_response}
```

### Flask Example

```python
from flask import Flask, request, jsonify
from integration import CBTIntegration

app = Flask(__name__)
cbt = CBTIntegration()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('question', '')
    
    if cbt.should_include_cbt(user_query):
        recommendations = cbt.cbt_kb.get_cbt_recommendation(user_query)
        response = cbt.cbt_kb.format_cbt_response(recommendations, user_query)
    else:
        response = "I understand your question. How can I help you further?"
    
    return jsonify({'answer': response})
```

## File Structure After Setup

```
CBT_System/
├── data_collector.py      # Data collection script
├── data_processor.py      # Data processing script
├── vectorizer.py          # Vector database creation
├── integration.py         # Main CBT integration API
├── setup.py              # Automated setup script
├── test_system.py        # System testing script
├── README.md             # Full documentation
├── QUICK_START.md        # This file
├── USAGE_GUIDE.md        # Detailed usage instructions
├── requirements.txt      # Python dependencies
└── cbt_setup.log         # Setup process log

cbt_data/                 # Created during setup
├── raw_data/            # Original collected data
├── structured_data/     # Processed CBT content
├── embeddings/          # Vector database files
├── collection.log       # Data collection log
├── processing.log       # Data processing log
└── vectorization.log    # Vector creation log
```

## Troubleshooting

### "CBT not available" error
```bash
# Re-run setup
python setup.py
```

### "Dependencies missing" error
```bash
# Install missing packages
pip install sentence-transformers faiss-cpu
```

### "No data collected" error
- Check internet connection
- Verify that public sources are accessible
- Re-run setup

### Performance issues
- Ensure you have at least 4GB free RAM
- Vector database creation requires significant memory
- Consider using a smaller embedding model for limited resources

## Next Steps

1. **Read Full Documentation**: See `README.md` for complete details
2. **Check Usage Examples**: See `USAGE_GUIDE.md` for more examples
3. **Integrate with Your App**: Use the integration examples above
4. **Test with Real Queries**: Try the system with actual user questions
5. **Monitor Performance**: Check logs for any issues

## Support

- Check log files for detailed error messages
- Review troubleshooting section above
- Ensure all dependencies are properly installed
- Re-run setup if issues persist

## Important Notes

### Ethical Use
- This system provides general CBT information only
- Always encourage users to seek professional help
- Include appropriate disclaimers in your application
- Do not use for crisis intervention or diagnosis

### Data Sources
- All data comes from public domain sources (NHS, NIMH, CCI Australia)
- No personal data is collected or stored
- Sources are properly attributed and licensed

### Performance
- Initial setup takes time but only needs to be done once
- Response generation is fast after setup
- Vector search is optimized for real-time use

That's it! You now have a complete CBT knowledge base system ready to enhance your mental health applications with evidence-based therapeutic techniques. 