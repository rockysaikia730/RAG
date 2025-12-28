# RAG: Retrieval-Augmented Generation for Argusa AI Challenge

A multimodal **Retrieval-Augmented Generation (RAG)** pipeline using sentence based chunking using LlamaIndex and Docling. This project integrates a backend for document processing and retrieval with **Streamlit** interface for querying your images, pdfs, emails and many other file types.

## 🚀 Features

* **Interactive UI**: A web interface built with `Streamlit` for easy interaction with your data.
* **Multimodality**: Supports multiple file types, images.
* **Precise**: Utilises metadata, augmented queries, indexing of nodes, vector search, keyword search for precise retrievals.

## 📂 Project Structure

```bash
RAG/
├── Experiments/       # Notebooks and scripts for testing RAG components
├── RAG/               # Core package containing retrieval and generation logic
├── logs/              # Application logs for debugging and monitoring
├── my_test_docs/      
├── StreamlitApp.py    # Main entry point for the Streamlit web application
├── exception.py       # Custom exception handling logic
├── logger.py          # Logging configuration
├── setup.py           # Package installation script
├── template.py        # Template utility script
└── requirements.txt   # List of Python dependencies

```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/rockysaikia730/RAG.git
cd RAG
```

2. **Create a Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate 

```

3. **Install Dependencies**
```bash
pip install -r requirements.txt

```

## 🏃 Usage

### 1. Run the Application

Launch the Streamlit interface:

```bash
streamlit run StreamlitApp.py

```

### 2. Chat with Your Data

* Open your browser at the provided local URL.
* Upload your pdfs and images.
* Enter your query in the chat input.
* The system will retrieve relevant context from your documents and generate an answer.
