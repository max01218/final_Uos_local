# CBT Integration System

A comprehensive Cognitive Behavioral Therapy (CBT) knowledge base and integration system for mental health applications.

## Overview

This system collects, processes, and integrates CBT resources from public domain sources to enhance mental health applications with evidence-based therapeutic techniques.

## Features

- **Data Collection**: Automated collection from government and academic sources
- **Data Processing**: Advanced text processing and categorization
- **Vector Database**: Fast semantic search using FAISS
- **CBT Integration**: Easy integration with existing systems
- **Evidence-Based**: Uses only verified public domain sources

## Quick Start

### 1. Install Dependencies

```bash
pip install requests beautifulsoup4 sentence-transformers faiss-cpu numpy
```

### 2. Run Setup

```bash
python setup.py
```

This will:
- Check dependencies
- Collect CBT data from public sources
- Process and clean the data
- Create vector embeddings
- Test the integration

### 3. Test Integration

```python
from integration import CBTIntegration

cbt = CBTIntegration()
status = cbt.get_cbt_status()
print(f"CBT Available: {status['available']}")
```

## System Components

### Data Collection (`data_collector.py`)
- Collects CBT resources from NHS, NIMH, and CCI Australia
- Respects robots.txt and rate limits
- Extracts clean content from web pages

### Data Processing (`data_processor.py`)
- Cleans and categorizes CBT content
- Extracts specific techniques and assessments
- Removes duplicates and low-quality content

### Vectorization (`vectorizer.py`)
- Creates sentence embeddings using Transformer models
- Builds FAISS index for fast similarity search
- Organizes metadata for efficient retrieval

### Integration (`integration.py`)
- Provides simple API for CBT enhancement
- Analyzes user queries for relevant techniques
- Formats professional responses with disclaimers

### Setup (`setup.py`)
- Orchestrates the complete setup process
- Validates dependencies and configuration
- Creates documentation and requirements

## CBT Techniques Included

### Cognitive Restructuring
- Thought challenging techniques
- Cognitive distortion identification
- Balanced thinking exercises

### Behavioral Activation
- Activity scheduling
- Pleasant event planning
- Behavioral experiments

### Exposure Therapy
- Systematic desensitization
- Fear hierarchy construction
- Gradual exposure protocols

### Problem-Solving Therapy
- Structured problem-solving steps
- Decision-making frameworks
- Coping strategy development

### Relaxation Techniques
- Deep breathing exercises
- Progressive muscle relaxation
- Mindfulness practices

### Psychoeducation
- Mental health education
- Symptom understanding
- Treatment information

## Usage Examples

### Basic Enhancement
```python
from integration import CBTIntegration

cbt = CBTIntegration()

# Enhance response with CBT techniques
enhanced_response = cbt.enhance_response_with_cbt(
    user_query="I feel anxious all the time",
    context="User has reported anxiety symptoms",
    base_response="I understand you're feeling overwhelmed"
)
```

### Direct CBT Recommendations
```python
# Get specific CBT recommendations
recommendations = cbt.cbt_kb.get_cbt_recommendation(
    "How can I challenge negative thoughts?"
)

formatted_response = cbt.cbt_kb.format_cbt_response(
    recommendations, 
    "How can I challenge negative thoughts?"
)
```

### Search CBT Techniques
```python
# Search for specific techniques
results = cbt.cbt_kb.search_cbt_techniques(
    "anxiety coping strategies",
    top_k=5
)
```

## Data Sources

All data is collected from public domain and open-access sources:

- **NHS (UK)**: Open Government License
- **NIMH (US)**: Public Domain
- **CCI Australia**: Creative Commons

## File Structure

```
CBT_System/
├── data_collector.py      # Data collection from public sources
├── data_processor.py      # Text processing and categorization
├── vectorizer.py          # Vector embedding creation
├── integration.py         # CBT integration API
├── setup.py              # Automated setup script
└── README.md             # This file

cbt_data/                 # Created during setup
├── raw_data/            # Original collected data
├── structured_data/     # Processed CBT content
└── embeddings/          # Vector database files
```

## Requirements

- Python 3.7+
- requests>=2.28.0
- beautifulsoup4>=4.11.0
- sentence-transformers>=2.2.0
- faiss-cpu>=1.7.0
- numpy>=1.21.0

## Quality Assurance

### Content Quality
- Source credibility scoring
- Technique diversity analysis
- Professional accuracy validation

### Safety Measures
- Professional disclaimers
- Crisis intervention protocols
- Appropriate therapeutic boundaries

### Ethical Considerations
- Public domain sources only
- Transparent data usage
- No personal data collection

## Performance

- Fast semantic search with FAISS
- Efficient embedding models
- Optimized for real-time responses
- Scalable architecture

## Troubleshooting

### Common Issues

**Dependencies not installed**
```bash
pip install -r requirements.txt
```

**CBT not available**
```bash
python setup.py
```

**Search returns no results**
- Check if vector database exists in `cbt_data/embeddings/`
- Verify data collection completed successfully

### Debug Commands

```bash
# Check system status
python -c "from integration import CBTIntegration; print(CBTIntegration().get_cbt_status())"

# Test search functionality
python integration.py

# Re-run setup
python setup.py
```

## Contributing

To extend the system:

1. Add new data sources in `data_collector.py`
2. Update technique categories in `data_processor.py`
3. Test new functionality thoroughly
4. Update documentation

## License

This project uses public domain and open-access sources. All code is provided for educational and research purposes.

## Support

For technical issues:
- Check log files for detailed error messages
- Review the troubleshooting section
- Verify system requirements
- Re-run setup if necessary

## Ethical Use

This system is designed to:
- Support mental health professionals
- Provide evidence-based information
- Maintain appropriate therapeutic boundaries
- Encourage professional consultation

It is NOT intended to:
- Replace professional therapy
- Provide crisis intervention
- Diagnose mental health conditions
- Offer medical advice

Always encourage users to seek professional help for mental health concerns. 