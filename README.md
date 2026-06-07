# Dynamic Knowledge Base Chatbot

A Python-based chatbot application that dynamically indexes and updates its vector database in response to changes in a local source folder. Powered by **ChromaDB**, **Sentence Transformers**, and **Streamlit** (with optional **Google Gemini** integration).

---

## Features
- **Dynamic Monitoring**: A background system automatically scans a folder for new, modified, or deleted files (`.txt`, `.md`, `.pdf`), updates their embeddings, and indexes them in real-time.
- **RAG (Retrieval-Augmented Generation)**: Uses a local Sentence Transformer model to encode user queries and match them semantically against document chunks.
- **Twin Execution Modes**:
  1. **Extractive Mode (Local & Free)**: Extracts and highlights the most relevant matching document snippets immediately, without requiring an external LLM API key.
  2. **Generative RAG Mode (Gemini)**: Uses Google Gemini models to construct conversational responses based solely on the retrieved documents.
- **Rich Dashboard**: Streamlit interface displaying indexed documents, chunk sizes, live background logs, and an option to upload files directly via the UI.

---

## Installation & Setup

1. **Verify Python Installation**
   Ensure you have Python 3.9+ installed on your computer.

2. **Install Dependencies**
   Navigate to the project directory and install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The first time you run the app, Sentence Transformers will download the `all-MiniLM-L6-v2` model to your local machine, which may take a few moments.)*

3. **Launch the Application**
   Run the Streamlit server to start the chatbot UI:
   ```bash
   streamlit run app.py
   ```

---

## How It Works Under the Hood

### 1. Ingestion & Chunking (`vector_store.py`)
- Reads file types like plain text, Markdown, and PDFs.
- Breaks down texts into semantic chunks of ~500 characters, overlapping contiguous segments by 50 characters to prevent context loss at splits.
- Splitting matches word boundaries so words are never sliced in half.

### 2. Embeddings (`embedder.py`)
- The text chunks are sent to the local `SentenceTransformer` model (`all-MiniLM-L6-v2`).
- Each chunk is converted into a 384-dimensional dense vector representing its semantic meaning.

### 3. Vector Database (`db/` & `vector_store.py`)
- Vector embeddings and original text chunks are stored in a local **ChromaDB** instance.
- Metadata is attached to each entry, including the source filename and the timestamp of ingestion.

### 4. Background Monitoring (`updater.py`)
- A background scanning thread runs every 10 seconds.
- It compares files in the `data/` folder against a JSON state manifest (`indexed_files.json`).
- If a file is **added** or **modified** (detected via size/modification-time mismatch), the updater chunks it, deletes its previous chunks in ChromaDB (if any), and adds the new ones.
- If a file is **deleted** from the folder, the updater automatically purges all of its associated chunks from ChromaDB, keeping the database perfectly in sync.
