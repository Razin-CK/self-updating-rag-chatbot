import logging
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Custom Embedding Function for ChromaDB using SentenceTransformers locally.
    It takes document texts and returns high-dimensional dense vector embeddings.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading embedding model '{self.model_name}'...")
        try:
            # Load the model. By default, it will download to cache on first run
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise e

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds the input documents using SentenceTransformers.
        Must conform to ChromaDB's EmbeddingFunction protocol.
        """
        try:
            # Ensure input is a list of strings
            if isinstance(input, str):
                input = [input]
            
            # Generate embeddings and convert them to a list of lists of floats
            logger.info(f"Generating embeddings for {len(input)} text chunks.")
            embeddings = self.model.encode(input, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise e
