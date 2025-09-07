# Legal Judgment Analysis System

A Natural Language Processing-based tool designed to generate structured summaries of construction court case judgments for better management of legal documents. This system employs rhetorical role classification and automated summarization to make complex legal judgments more accessible to construction professionals.

## Overview

This research addresses the gap between legal expertise and construction practice by developing an automated approach to legal judgment summarization. The tool uses a fine-tuned InLegalBERT model to identify rhetorical roles in legal documents and generates structured summaries using large language models.

### Key Features

- **Rhetorical Role Classification**: Automatically identifies and labels sentences with rhetorical roles (Facts, Arguments, Precedent, etc.)
- **Automated Summarization**: Generates comprehensive summaries using state-of-the-art language models
- **Custom Summary Generation**: Create tailored summaries by selecting specific rhetorical roles
- **Court Case Search**: Integrated search functionality for Indian Kanoon database
- **User-Friendly Interface**: Web-based interface built with Gradio
- **Multiple Output Formats**: Download results in various formats (text, markdown)

## Technical Architecture

### Rhetorical Role Classification

The system identifies nine key rhetorical roles in legal judgments: - **Facts**: Background information and events leading to the dispute - **Issues**: Questions of law or facts to be determined by the court - **Arguments**: Contentions made by the parties - **Statute**: References to relevant laws, regulations, or codes - **Precedent**: Citations of previous court decisions - **Ruling by Lower Court**: Decisions made by courts of lower instance - **Discussion by the Court**: The court's analysis and interpretation - **Ratio of the Decision**: The legal principle or reasoning behind the judgment - **Ruling by Present Court**: The final decision and orders

### Model Performance

- **InLegalBERT Model**: Achieved 68% accuracy in rhetorical role identification
- **Summarization**: Comparative evaluation showed Gemma-3-27B outperformed Llama-3.2-3B across multiple metrics
- **Domain-Specific Training**: Model fine-tuned on 30 Supreme Court judgments from Indian construction disputes

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM

### Dependencies Installation

Using Conda (Recommended): `bash conda env create -f environment.yml conda activate legal-judgment-analysis`

Using pip: `bash pip install -r requirement.txt`

### Required Python Packages

- `gradio>=3.50.0` - Web interface framework
- `transformers>=4.30.0` - Hugging Face transformers library
- `torch>=2.0.0` - PyTorch deep learning framework
- `spacy>=3.6.0` - Natural language processing library
- `PyMuPDF` - PDF text extraction
- `openai` - OpenRouter API integration
- `requests` - HTTP requests for Indian Kanoon API
- `reportlab` - PDF generation
- en-core-web-lg model from [https://spacy.io/models/en](https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.8.0)
- `scikit-learn` - Fine tuning Rhetorical role annotation model
  
### Model Setup

The `trained_model` directory should contain the fine-tuned InLegalBERT model files: - `config.json` - `pytorch_model.bin` - `tokenizer.json` - `tokenizer_config.json` - `vocab.txt`

## Usage

### Starting the Application

`bash python app.py`

The application will launch a web interface accessible at `http://localhost:7860`

### Basic Workflow

1. **API Configuration**: Enter your OpenRouter API key for LLM access
2. **Document Upload**: Upload a PDF of a legal judgment
3. **Analysis**: Click "Analyze Document" to process the file
4. **Review Results**:
5. View annotated text with rhetorical roles
6. Read the generated summary
7. Create custom summaries with selected roles
8. Download results in various formats

### Advanced Features

#### Court Case Search

- Search the Indian Kanoon database directly within the application
- Filter by document type, date range, and other criteria
- Preview and download cases for analysis

#### Custom Summary Generation

- Select specific rhetorical roles to include
- Adjust summary length (100-500 words)
- Generate focused summaries for specific use cases

## API Configuration

### OpenRouter API Key

The system uses OpenRouter for accessing various language models. Obtain your API key from [OpenRouter](https://openrouter.ai/) and configure it in the application. Selection of Large Language Model can be made based on user's choice and requirements.

For this research following models were used: `google/gemma-3-27b-it` and `meta-llama/llama-3.2-3b-instruct`

### Indian Kanoon API (Optional)

For court case search functionality, obtain an API token from [Indian Kanoon](https://indiankanoon.org/api/).


## Research Background

This tool is based on research conducted as part of an M.Tech Final Year Project focusing on improving decision-making capabilities of contract managers using pre-existing legal information, specifically Supreme Court case judgments in India.

### Academic Contributions

- **Novel Approach**: First system to combine rhetorical role classification with LLM-based summarization for Indian construction law
- **Domain-Specific Model**: Fine-tuned InLegalBERT specifically for construction-related legal judgments
- **Practical Application**: Bridges the gap between legal expertise and construction practice
- **Performance Validation**: Comprehensive evaluation using multiple metrics (ROUGE, BLEU, METEOR, BERTScore)

### Research Objectives

1. **Identify Pre-existing Information**: Establish case laws as valuable resources for construction professionals
2. **Develop Accessible Methods**: Create automated summarization to make legal judgments accessible
3. **Build Practical Tools**: Develop a structured model leveraging legal precedents for improved decision-making

## Limitations

### Current Limitations

- **Model Accuracy**: 68% accuracy in rhetorical role identification leaves room for improvement
- **Dataset Size**: Training on 30 judgments, while theoretically justified, could benefit from expansion
- **Domain Scope**: Currently focused on Indian construction law
- **Language Support**: Limited to English legal documents

## Citation

`**Paper Citation**`

## License

This project is released under the MIT License. See LICENSE file for details.
