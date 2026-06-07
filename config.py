import os

# Base directory for the chatbot project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "db")

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ChromaDB collection name
CHROMA_COLLECTION_NAME = "dynamic_knowledge_base"

# Embedding model settings
# Using a lightweight but highly capable general sentence transformer model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Cross-Encoder Re-ranker model settings
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RE_RANK_CANDIDATES = 10  # First-stage candidates to fetch
TOP_K_RESULTS = 3         # Final number of chunks to keep

# Chunking settings
CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 50  # Overlap between contiguous chunks

# Scanner settings
SCAN_INTERVAL_SECONDS = 10  # Check for new files every 10 seconds in the background
TRACKING_FILE = os.path.join(DB_DIR, "indexed_files.json")
