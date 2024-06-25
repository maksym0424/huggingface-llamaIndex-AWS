import faiss
from typing import List, Optional
from pathlib import Path
from .base import BaseVectorStore
from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
    Document
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.faiss import FaissVectorStore

class MyFaissVectorStore(BaseVectorStore):
    """
    FAISS Vector Store
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index_path = "./indexes/faiss"
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        
    def load_index(self):
        if not self.index_path:
            return None
        file_path = Path(f"{self.index_path}")
        # print(file_path)
        if file_path.exists() == False:
            return None
        
        vector_store = FaissVectorStore.from_persist_dir(self.index_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=self.index_path
        )
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        return index
    
    def add_document(self, documents: List[Document] = []):
        index = self.load_index()
        if not index:
            vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)    
            
            index = VectorStoreIndex.from_documents(
                documents, embed_model=self.embed_model, storage_context=storage_context
            )
        else:
            for document in documents:
                index.insert(document=document)
        
        index.storage_context.persist(self.index_path)
        print("Documents embedded successfully.")
        
    def retreive(self, query: str):
        index = self.load_index()
        
        if not index:
            return None
        
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3, embed_model=self.embed_model)
        response = retriever.retrieve(query)
        return response