# Enhanced CBT Data System v2.0

A comprehensive, advanced Cognitive Behavioral Therapy (CBT) data collection, processing, and vectorization system with multi-source integration, intelligent classification, and hierarchical indexing.

## Overview

The Enhanced CBT System extends the original CBT integration with:

- **Expanded Data Sources**: 10+ authoritative sources (APA, WHO, NICE, etc.)
- **Intelligent Processing**: Advanced deduplication, quality assessment, and classification
- **Multi-level Vectorization**: Hierarchical indices with quality tiers and category-specific search
- **Quality Control**: Comprehensive content assessment and validation
- **Flexible Configuration**: JSON-based configuration management

## Quick Start

### 1. Install Dependencies

```bash
pip install requests beautifulsoup4 sentence-transformers faiss-cpu numpy scikit-learn nltk torch
```

### 2. Run Enhanced Setup

```bash
cd CBT_System
python enhanced_setup.py --action full
```

This will execute the complete pipeline:
- Collect data from 10+ authoritative sources
- Process and classify content intelligently
- Create hierarchical vector indices
- Generate comprehensive system report

### 3. Quick Update (Priority Sources Only)

```bash
python enhanced_setup.py --action quick
```

## System Components

### Enhanced Data Collector (`enhanced_data_collector.py`)

**New Features:**
- **Expanded Sources**: Government, professional organizations, international bodies, clinical guidelines
- **Quality Assessment**: Real-time content quality scoring during collection
- **Structured Extraction**: Automatic identification of techniques, steps, and concepts
- **Rate Limiting**: Respectful data collection with configurable delays
- **Content Deduplication**: Hash-based duplicate detection during collection

**Supported Sources:**
- NHS (UK National Health Service)
- NIMH (US National Institute of Mental Health)
- CCI Australia (Centre for Clinical Interventions)
- APA (American Psychological Association)
- WHO (World Health Organization)
- NICE (UK National Institute for Health and Care Excellence)
- CAMH (Centre for Addiction and Mental Health)
- Mental Health America

### Enhanced Data Processor (`enhanced_data_processor.py`)

**New Features:**
- **Advanced Deduplication**: Semantic similarity detection beyond exact matches
- **Intelligent Classification**: Automatic categorization into CBT technique types
- **Quality Scoring**: Multi-factor quality assessment including readability, structure, and professional content
- **Structure Extraction**: Automatic extraction of steps, key points, and examples
- **Content Type Recognition**: Identification of technique descriptions, worksheets, assessments, etc.

**Classification Categories:**
- Cognitive Restructuring
- Behavioral Activation
- Exposure Therapy
- Problem Solving
- Relaxation Techniques
- Psychoeducation

### Enhanced Vectorizer (`enhanced_vectorizer.py`)

**New Features:**
- **Multiple Embedding Models**: Support for different models optimized for various use cases
- **Vectorization Strategies**: Content-based, technique-focused, structured-steps, key-concepts
- **Hierarchical Indexing**: Quality-tiered indices (Premium/Standard/Basic)
- **Category-Specific Indices**: Separate indices for each CBT technique category
- **Performance Optimization**: Automatic index optimization based on dataset size

**Quality Tiers:**
- **Premium (85+ score)**: Highest quality, evidence-based content
- **Standard (70-84 score)**: Good quality, reliable content
- **Basic (60-69 score)**: Acceptable quality, general content

## Configuration

The system uses `enhanced_config.json` for comprehensive configuration:

```json
{
  "data_collection": {
    "priority_filter": 1,
    "source_configuration": {
      "government_sources": {"enabled": true, "priority": 1},
      "professional_orgs": {"enabled": true, "priority": 2}
    }
  },
  "data_processing": {
    "deduplication": {"semantic_similarity_threshold": 0.85},
    "quality_assessment": {"minimum_enhanced_score": 60}
  },
  "vectorization": {
    "models": {"primary": "all-MiniLM-L6-v2"},
    "strategies": ["content_based", "technique_focused"]
  }
}
```

## Usage Examples

### Basic Pipeline Execution

```python
from enhanced_setup import EnhancedCBTSystemSetup

# Initialize system
setup = EnhancedCBTSystemSetup("cbt_data")

# Run complete pipeline
success = setup.run_full_pipeline()
```

### Individual Component Usage

```python
# Data Collection Only
from enhanced_data_collector import EnhancedCBTDataCollector

collector = EnhancedCBTDataCollector()
results = collector.collect_all_enhanced_sources(priority_filter=1)
```

```python
# Data Processing Only
from enhanced_data_processor import EnhancedCBTDataProcessor

processor = EnhancedCBTDataProcessor()
output_path = processor.process_all_enhanced_data()
```

```python
# Vectorization Only
from enhanced_vectorizer import EnhancedCBTVectorizer

vectorizer = EnhancedCBTVectorizer()
result = vectorizer.vectorize_enhanced_data("clinical", "technique_focused")
```

### Integration with Existing System

```python
from integration import CBTIntegration

# Use enhanced data with existing integration
cbt = CBTIntegration(base_dir="cbt_data")
status = cbt.get_cbt_status()
results = cbt.cbt_kb.search_cbt_techniques("anxiety coping", top_k=5)
```

## Command Line Interface

### Available Actions

```bash
# Complete pipeline
python enhanced_setup.py --action full

# Quick update (priority sources only)
python enhanced_setup.py --action quick

# Individual components
python enhanced_setup.py --action collect
python enhanced_setup.py --action process
python enhanced_setup.py --action vectorize

# System report
python enhanced_setup.py --action report

# Integration test
python enhanced_setup.py --action test

# Custom configuration
python enhanced_setup.py --action full --config custom_config.json
```

### Custom Base Directory

```bash
python enhanced_setup.py --action full --base-dir /path/to/custom/directory
```

## Output Structure

```
cbt_data/
├── raw_data/
│   ├── government/           # NHS, NIMH, government sources
│   ├── professional_orgs/    # APA, professional organizations
│   ├── international_orgs/   # WHO, international bodies
│   ├── clinical_guidelines/  # NICE, clinical guidelines
│   ├── academic/            # Academic sources (optional)
│   ├── nonprofit/           # Nonprofit organization sources
│   └── processed/           # Enhanced processed data
├── structured_data/
│   ├── techniques/          # Categorized CBT techniques
│   ├── assessments/         # Assessment tools
│   ├── worksheets/          # Therapeutic worksheets
│   └── protocols/           # Treatment protocols
├── embeddings/
│   ├── cbt_index_premium_*.faiss    # High-quality index
│   ├── cbt_index_standard_*.faiss   # Standard quality index
│   ├── cbt_index_category_*.faiss   # Category-specific indices
│   └── cbt_index_summary.json       # Index metadata
├── reports/
│   └── system_report_*.json         # System status reports
└── logs/
    └── enhanced_setup_*.log          # Detailed logs
```

## Quality Metrics

The system tracks various quality metrics:

### Collection Metrics
- Sources successfully accessed
- Content items collected per source
- Average content quality score
- Content type distribution

### Processing Metrics
- Deduplication statistics (exact + semantic)
- Classification confidence scores
- Quality score distribution
- Structure extraction success rate

### Vectorization Metrics
- Embedding creation success rate
- Index optimization results
- Search functionality validation
- Performance benchmarks

## Integration with FastAPI Server

The enhanced system is fully compatible with the existing FastAPI server:

```python
# In fastapi_server.py, the CBT integration will automatically
# use the enhanced data if available
from CBT_System.integration import CBTIntegration

cbt_integration = CBTIntegration(base_dir="CBT_System/cbt_data")
```

## Monitoring and Maintenance

### System Reports

Generate comprehensive system reports:

```bash
python enhanced_setup.py --action report
```

Reports include:
- Data collection status and statistics
- Processing quality metrics
- Vectorization index information
- Recommendations for system improvements

### Log Management

The system generates detailed logs for:
- Data collection activities
- Processing decisions and quality assessments
- Vectorization progress and results
- Integration test results

### Automatic Cleanup

Configure automatic maintenance in `enhanced_config.json`:

```json
{
  "maintenance": {
    "automatic_cleanup": {
      "enabled": true,
      "remove_old_logs": true,
      "days_to_keep_logs": 30
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install sentence-transformers faiss-cpu scikit-learn nltk
   ```

2. **NLTK Data Not Found**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Memory Issues**
   ```json
   {
     "performance": {
       "memory_management": {
         "max_memory_usage_gb": 4
       }
     }
   }
   ```

4. **Network Issues**
   ```json
   {
     "data_collection": {
       "rate_limiting": {
         "delay_between_sources": 10,
         "timeout": 60
       }
     }
   }
   ```

### Validation Commands

```bash
# Test system integration
python enhanced_setup.py --action test

# Check dependencies
python -c "from enhanced_setup import EnhancedCBTSystemSetup; setup = EnhancedCBTSystemSetup(); print(setup.check_dependencies())"
```

## Performance Optimization

### GPU Acceleration

Enable GPU acceleration for vectorization:

```json
{
  "performance": {
    "optimization": {
      "use_gpu_if_available": true
    }
  }
}
```

### Batch Processing

Optimize for large datasets:

```json
{
  "vectorization": {
    "batch_processing": {
      "batch_size": 64,
      "show_progress": true
    }
  }
}
```

### Index Optimization

The system automatically optimizes indices based on dataset size:
- Small datasets (<100): Flat index
- Medium datasets (100-1000): IVF index
- Large datasets (>1000): IVF-PQ index

## Development and Extension

### Adding New Data Sources

1. Edit `enhanced_data_collector.py`
2. Add source configuration to `self.sources`
3. Update `enhanced_config.json` if needed
4. Test with `--action collect`

### Custom Quality Metrics

1. Modify `enhanced_data_processor.py`
2. Update `calculate_enhanced_quality_score()`
3. Add new criteria to configuration

### New Vectorization Strategies

1. Edit `enhanced_vectorizer.py`
2. Add strategy to `extract_vectorization_text()`
3. Update configuration file

## Support and Contribution

For issues, feature requests, or contributions:
1. Check system logs in `cbt_data/logs/`
2. Run system report for diagnostics
3. Verify configuration settings
4. Test individual components

## License and Usage

This enhanced system builds upon the original CBT integration and maintains the same licensing approach for educational and research purposes. All data sources are accessed according to their respective license agreements.

---

**Version**: 2.0.0  
**Compatibility**: Original CBT Integration System  
**Requirements**: Python 3.7+, 8GB+ RAM recommended 