
---

# 🧠 RAGer — LangChain + LangGraph Retrieval-Augmented Generation System

RAGer is a **Retrieval-Augmented Generation (RAG)** pipeline built using **LangChain**, **LangGraph**, and **Chroma**.
It allows you to load PDF documents, embed them using Hugging Face models, and interact with them using an intelligent LLM (OpenAI GPT-4o-mini).
The system also performs **relevancy checking**, ensuring responses are based only on the provided documents.

---

## 🚀 Features

* 📄 Loads and processes PDF documents from the `data/` directory
* 🔍 Embeds text using `sentence-transformers/all-MiniLM-L6-v2`
* 🧠 Stores embeddings in a persistent **Chroma** vector database
* 🤖 Uses **LangGraph** to control workflow between LLM, tools, and memory
* 🧩 Includes a **retriever tool** that the LLM can invoke to fetch document context
* ⚙️ Performs **relevancy checking** before answering
* 💬 Interactive CLI interface for asking questions

---

## 🧰 Tech Stack

* [LangChain](https://python.langchain.com/)
* [LangGraph](https://github.com/langchain-ai/langgraph)
* [Chroma](https://www.trychroma.com/)
* [OpenAI GPT models](https://platform.openai.com/)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/RAGer.git
cd RAGer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📂 Directory Structure

```
RAGer/
│
├── RAGer.py                # Main application script
├── requirements.txt        # Dependencies
├── data/                   # Place your PDF documents here
├── chroma_langchain_db/    # Auto-generated vector database
└── .env                    # API keys (not committed)
```

---

## 🧑‍💻 Usage

Run the script directly:

```bash
python RAGer.py
```

You’ll be prompted to enter a question in the terminal:

```
Enter the question: What are the rules for trading properties in Monopoly?
```

The assistant will:

1. Check if your question is relevant (Monopoly, Chess, UNO documents)
2. Retrieve information from the vectorstore
3. Generate an accurate, document-grounded answer

---

## 🧩 How It Works

1. **Document Loading:**
   Loads all `.pdf` files from the `data/` directory using `DirectoryLoader`.

2. **Text Splitting & Embedding:**
   Splits documents into chunks and embeds them with `HuggingFaceEmbeddings`.

3. **Vector Storage:**
   Stores embeddings in a persistent Chroma database.

4. **Retriever Tool:**
   The LLM uses this tool to look up relevant context dynamically.

5. **LangGraph Workflow:**
   Orchestrates message flow:

   * Relevancy check
   * LLM call
   * Tool execution
   * Final response

---

## 🧠 Example Questions

```
What are the winning conditions in UNO?
How does castling work in Chess?
Can you mortgage properties in Monopoly?
```

---

## ⚠️ Notes

* Only questions related to **Monopoly**, **Chess**, or **UNO** will be answered.
* If you add new PDFs, delete the existing `chroma_langchain_db/` folder to rebuild embeddings.
* Make sure your API keys are valid before running the script.

---


