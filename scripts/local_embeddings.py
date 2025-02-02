import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalEmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with CUDA if available."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing local embedding model on {device}")
            self._model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            if device == 'cuda':
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.error(f"Failed to initialize local embedding model: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts using the local model.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embeddings as float lists, or None if generation fails
        """
        try:
            # Convert texts to embeddings
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=32
            )
            
            # Convert to numpy and then to list for consistency with OpenAI format
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            return None 