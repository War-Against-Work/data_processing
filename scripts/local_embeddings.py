import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    _instance = None
    _model = None
    _tokenizer = None
    
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
            
            # Initialize model and tokenizer
            model_name = "BAAI/bge-large-en-v1.5"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            
            # Move model to GPU if available
            self._model = self._model.to(device)
            self._device = device
            
            if device == 'cuda':
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
            # Set model to evaluation mode
            self._model.eval()
            
        except Exception as e:
            logger.error(f"Failed to initialize local embedding model: {e}")
            raise

    def _batch_encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts in batches to manage memory."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self._device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self._model(**encoded_input)
                # Use attention mask for mean pooling
                attention_mask = encoded_input['attention_mask']
                embeddings = self._mean_pooling(model_output, attention_mask)
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
            all_embeddings.append(embeddings)
            
        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to create sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts using the local model.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embeddings as float lists, or None if generation fails
        """
        try:
            # Get base embeddings (1024 dim)
            embeddings = self._batch_encode(texts)
            
            # Project to 1536 dimensions using a learned linear transformation
            # For now, we'll use a simple zero-padding approach
            padded_embeddings = F.pad(embeddings, (0, 1536 - embeddings.size(1)))
            
            # Convert to numpy and then to list for consistency with OpenAI format
            return padded_embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            return None 