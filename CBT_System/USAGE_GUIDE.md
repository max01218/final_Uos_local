# CBT System Usage Guide

## Setup Complete

Your CBT (Cognitive Behavioral Therapy) system has been successfully set up and integrated.

## System Components

1. **Data Collection**: Collected CBT resources from public domain sources
2. **Data Processing**: Cleaned and categorized CBT techniques and content
3. **Vectorization**: Created searchable vector database of CBT knowledge
4. **Integration**: Ready to enhance responses with CBT techniques

## Using the CBT System

### Basic Integration:

```python
from integration import CBTIntegration

# Initialize CBT integration
cbt_integration = CBTIntegration()

# Enhance responses with CBT techniques
enhanced_response = cbt_integration.enhance_response_with_cbt(
    user_query="I feel anxious all the time",
    context="relevant context",
    base_response="base response"
)
```

### Testing CBT functionality:

```bash
python integration.py
```

## CBT Features Available

- Cognitive restructuring techniques
- Behavioral activation strategies  
- Exposure therapy guidance
- Problem-solving approaches
- Relaxation techniques
- Psychoeducational content

## Files Created

- `cbt_data/`: Main CBT data directory
- `cbt_data/embeddings/`: Vector database files
- `cbt_data/structured_data/`: Organized CBT content
- `cbt_setup.log`: Setup process log

## Next Steps

1. Integrate CBT enhancement into your existing system endpoints
2. Test with real user queries
3. Monitor CBT technique recommendations
4. Update data periodically by re-running collection scripts

## Troubleshooting

If CBT features are not working:
1. Check that all dependencies are installed
2. Verify that vector database files exist in `cbt_data/embeddings/`
3. Check logs for error messages
4. Re-run setup if needed: `python setup.py`

## Support

For technical issues, check the logs:
- `cbt_setup.log`: Overall setup process
- `cbt_data/collection.log`: Data collection issues  
- `cbt_data/processing.log`: Data processing issues
- `cbt_data/vectorization.log`: Vector database issues
