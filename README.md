# # Automated File Organizer utilizing AI

An intelligent document management tool that automatically organizes files based on semantic content using machine learning and natural language processing.

## Features

- **Multi-format Support**: PDF, DOCX, PPTX, TXT, Excel files
- **Smart Clustering**: Supervised (custom folders) and unsupervised (automatic) organization
- **Semantic Understanding**: Uses NLP embeddings to understand document meaning beyond keywords
- **Dual Interface**: GUI and CLI options
- **Visualization**: Interactive cluster analysis with t-SNE plotting

## Installation

```bash
git clone https://github.com/stillkrish/Automated-File-Organizer.git
cd Automated-File-Organizer
pip install -r requirements.txt
```

### Dependencies

```
numpy
pandas
scikit-learn
matplotlib
nltk
sentence-transformers
PyMuPDF
python-docx
openpyxl
python-pptx
```

## Usage

### GUI Mode

```bash
python main.py
```

### CLI Mode

```bash
python cli.py
```

## How It Works

1. **Text Extraction**: Extracts and preprocesses text from multiple file formats
2. **Embedding**: Converts documents to semantic vectors using Sentence Transformers
3. **Clustering**: Groups similar documents using K-means with silhouette score optimization
4. **Organization**: Moves files to semantically meaningful folders

### Clustering Methods

**Automatic**: Algorithm determines optimal clusters and generates folder names

**Custom**: User defines categories, documents assigned via cosine similarity

## Technical Stack

- **TextProcessor**: Multi-format text extraction and cleaning
- **TextEmbedder**: Semantic vector generation ("all-MiniLM-L6-v2" model)
- **ClusterMaker**: K-means clustering with automatic optimization
- **Interface**: Tkinter GUI and command-line options

## Requirements

- Python 3.7+
- 512MB+ RAM recommended for processing large document sets
