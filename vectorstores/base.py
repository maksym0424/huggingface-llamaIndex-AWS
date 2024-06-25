from typing import Optional, Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
class BaseVectorStore:
    """
    A class representing a vector store.
    """
    
    def __init__(self, index_path: Optional[str] = None, embed_model: Optional[Any] = None, dimension: Optional[int] = 1024):
        self.index_path = index_path

        self.dimension = dimension
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")