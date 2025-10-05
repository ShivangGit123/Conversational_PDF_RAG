# 📄 Conversational RAG with PDF using Groq, LangChain, and Streamlit

This is a simple and powerful Streamlit application that allows you to upload any PDF file and interact with it conversationally. It uses **LangChain's Retrieval-Augmented Generation (RAG)** pipeline and Groq's **Gemma-2B-IT** model for fast, open-source LLM responses. HuggingFace embeddings and Chroma are used to semantically understand the PDF content.

---

## 🚀 Features

- 📄 Upload and parse PDF files
- 🔍 Ask context-aware questions based on PDF content
- 🧠 Conversational memory for multi-turn dialogue
- ⚡ Uses Groq’s blazing-fast inference for open-source LLMs
- 🧩 Modular RAG pipeline with LangChain
- 💬 Streamlit interface for easy interaction

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq API](https://console.groq.com/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Chroma Vector Store](https://www.trychroma.com/)
- [pdfminer.six](https://pypi.org/project/pdfminer.six/)

---

## 📦 Installation

### 1. Clone the repository
git clone https://github.com/ShivangGit123/conversational-rag-pdf.git
cd conversational-rag-pdf

### 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### 3.Install dependencies
pip install -r requirements.txt

### 4.Create a .env file
HF_TOKEN=your_huggingface_token

---
▶️ Running the App
streamlit run app.py

---


