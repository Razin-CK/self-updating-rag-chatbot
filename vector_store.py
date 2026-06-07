import os
import time
import logging
import re
from typing import List, Dict, Any, Tuple
import chromadb
from pypdf import PdfReader
from sentence_transformers import CrossEncoder
from embedder import LocalSentenceTransformerEmbeddingFunction
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Handles all interactions with ChromaDB, including file ingestion,
    chunking, metadata tracking, deletion, querying, re-ranking, and resets.
    """
    def __init__(self):
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=config.DB_DIR)
        
        # Initialize our custom Sentence Transformer embedder
        self.embedding_function = LocalSentenceTransformerEmbeddingFunction()
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
        logger.info(f"Connected to ChromaDB collection: '{config.CHROMA_COLLECTION_NAME}'")
        
        # Lazily loaded Cross-Encoder for second-stage re-ranking
        self.cross_encoder = None

    def _init_cross_encoder(self):
        """
        Lazily loads the Cross-Encoder model only when required.
        """
        if self.cross_encoder is None:
            logger.info(f"Loading Cross-Encoder model '{config.CROSS_ENCODER_MODEL_NAME}'...")
            try:
                self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)
                logger.info("Cross-Encoder model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading Cross-Encoder model: {e}")
                raise e

    @staticmethod
    def read_file(file_path: str) -> str:
        """
        Reads files based on extension (.txt, .md, .pdf) and returns string content.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
        """
        Advanced recursive sentence-bound splitter:
        1. Splits by paragraph first.
        2. If a paragraph fits inside the chunk_size, it keeps it together.
        3. If not, it splits that paragraph into sentences.
        4. Keeps semantic units together without splitting sentences in half.
        """
        if not text or not text.strip():
            return []
            
        # Split text by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0
        
        for paragraph in paragraphs:
            # Paragraph fits in the current chunk
            if len(paragraph) < chunk_size:
                if current_len + len(paragraph) + 2 <= chunk_size:
                    current_chunk.append(paragraph)
                    current_len += len(paragraph) + 2
                else:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    
                    # Create overlap by keeping some trailing items
                    overlap_items = []
                    overlap_len = 0
                    for item in reversed(current_chunk):
                        if overlap_len + len(item) + 2 <= overlap:
                            overlap_items.insert(0, item)
                            overlap_len += len(item) + 2
                        else:
                            break
                    current_chunk = overlap_items + [paragraph]
                    current_len = overlap_len + len(paragraph) + 2
            else:
                # Paragraph is too long; split into sentences
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
                for sentence in sentences:
                    if current_len + len(sentence) + 1 <= chunk_size:
                        current_chunk.append(sentence)
                        current_len += len(sentence) + 1
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            
                        # Create overlap by keeping trailing sentences
                        overlap_items = []
                        overlap_len = 0
                        for item in reversed(current_chunk):
                            if overlap_len + len(item) + 1 <= overlap:
                                overlap_items.insert(0, item)
                                overlap_len += len(item) + 1
                            else:
                                break
                        current_chunk = overlap_items + [sentence]
                        current_len = overlap_len + len(sentence) + 1
                        
        if current_chunk:
            if len(current_chunk) == 1 and current_chunk[0] == "":
                pass
            else:
                chunks.append("\n\n".join(current_chunk))
                
        return [c for c in chunks if c.strip()]

    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        Reads, chunks, embeds, and indexes a file.
        If the file was already indexed, it deletes existing chunks first to prevent duplicates.
        """
        filename = os.path.basename(file_path)
        logger.info(f"Processing document for indexing: {filename}")
        
        # 1. Read document text
        try:
            text = self.read_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            return {"status": "error", "message": f"Read failed: {e}"}

        # 2. Chunk text using advanced recursive sentence chunker
        chunks = self.chunk_text(text)
        if not chunks:
            logger.warning(f"No text extracted or chunked from {filename}.")
            return {"status": "warning", "message": "Document empty or no extractable text."}
            
        # 3. Handle updates (delete previous records of this file)
        self.delete_document(filename)

        # 4. Ingest new chunks
        ids = []
        metadatas = []
        documents = []
        
        timestamp = time.time()
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": filename,
                "chunk_index": idx,
                "timestamp": timestamp,
                "total_chunks": len(chunks)
            })

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Indexed {len(chunks)} chunks for {filename}.")
            return {
                "status": "success", 
                "chunks_added": len(chunks),
                "timestamp": timestamp
            }
        except Exception as e:
            logger.error(f"Failed to index {filename} into ChromaDB: {e}")
            return {"status": "error", "message": f"Index failed: {e}"}

    def delete_document(self, filename: str):
        """
        Deletes all chunks associated with a specific file from ChromaDB.
        """
        try:
            self.collection.delete(where={"source": filename})
            logger.info(f"Deleted old chunks (if any) for document: {filename}")
        except Exception as e:
            logger.error(f"Error deleting old database chunks for {filename}: {e}")

    def query(self, query_text: str, use_reranking: bool = False, n_results: int = config.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for similar document chunks.
        Optionally uses a Cross-Encoder to re-rank the initial candidate hits.
        """
        try:
            # 1. Determine how many candidates to fetch
            # If re-ranking, fetch more candidates (e.g., M=10) first.
            fetch_count = config.RE_RANK_CANDIDATES if use_reranking else n_results
            
            # Retrieve stats to check total available chunks
            total_chunks = self.collection.count()
            if total_chunks == 0:
                return []
                
            fetch_count = min(fetch_count, total_chunks)
            
            # 2. Run vector search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=fetch_count
            )
            
            hits = []
            if not results or not results["documents"] or len(results["documents"][0]) == 0:
                return hits
                
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]
            
            for i in range(len(docs)):
                hits.append({
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i],
                    "rerank_score": None  # Populated only if re-ranked
                })
                
            # 3. Two-Stage Re-ranking (Cross-Encoder)
            if use_reranking and hits:
                # Initialize Cross-Encoder model
                self._init_cross_encoder()
                
                # Format pairs: [[Query, Document1], [Query, Document2], ...]
                pairs = [[query_text, hit["text"]] for hit in hits]
                
                # Generate Cross-Encoder scores (higher is more relevant)
                scores = self.cross_encoder.predict(pairs)
                
                # Assign scores
                for hit, score in zip(hits, scores):
                    hit["rerank_score"] = float(score)
                    
                # Sort descending by re-rank score
                hits.sort(key=lambda x: x["rerank_score"], reverse=True)
                
                # Keep top K results
                hits = hits[:n_results]
                
            return hits
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    def wipe_database(self) -> Dict[str, Any]:
        """
        Wipes the entire vector database:
        1. Deletes the ChromaDB collection.
        2. Deletes the indexed manifest file.
        3. Clears all files in the docs/ directory.
        """
        logger.warning("Wiping vector database and docs folder...")
        try:
            # Delete Chroma collection
            self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
            
            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            
            # Clear manifest tracking file
            if os.path.exists(config.TRACKING_FILE):
                os.remove(config.TRACKING_FILE)
                
            # Clear all files inside docs directory
            files_deleted = []
            if os.path.exists(config.DOCS_DIR):
                for filename in os.listdir(config.DOCS_DIR):
                    file_path = os.path.join(config.DOCS_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_deleted.append(filename)
                        
            logger.info("Database and docs folder cleared successfully.")
            return {"status": "success", "files_cleared": files_deleted}
        except Exception as e:
            logger.error(f"Error wiping database: {e}")
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves database collection statistics.
        """
        try:
            count = self.collection.count()
            
            # Fetch all metadata to identify unique files
            all_data = self.collection.get(include=["metadatas"])
            unique_files = set()
            file_chunk_counts = {}
            
            if all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    source = meta.get("source", "unknown")
                    unique_files.add(source)
                    file_chunk_counts[source] = file_chunk_counts.get(source, 0) + 1
            
            return {
                "total_chunks": count,
                "total_files": len(unique_files),
                "files": list(unique_files),
                "file_chunk_counts": file_chunk_counts
            }
        except Exception as e:
            logger.error(f"Error getting collection statistics: {e}")
            return {"total_chunks": 0, "total_files": 0, "files": [], "file_chunk_counts": {}}
