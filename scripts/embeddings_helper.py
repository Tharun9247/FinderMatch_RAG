#!/usr/bin/env python3
"""
Hugging Face Embeddings Helper
Provides embedding functionality using Hugging Face Inference API with fallback to TF-IDF.
"""

import os
import requests
import json
import warnings
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import InferenceClient


class HuggingFaceEmbeddings:
    """Hugging Face embeddings with fallback to TF-IDF"""
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the embeddings helper.
        
        Args:
            api_token: Hugging Face API token. If None, will try to get from HF_API_TOKEN env var.
        """
        self.api_token = api_token or os.getenv('HF_API_TOKEN')
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.use_hf_api = bool(self.api_token)
        
        # Initialize the Hugging Face client
        if self.use_hf_api:
            try:
                self.client = InferenceClient(
                    model=self.model_name,
                    token=self.api_token
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize Hugging Face client: {e}")
                self.use_hf_api = False
                self.client = None
        else:
            self.client = None
        
        # TF-IDF fallback components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_documents = []
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text using Hugging Face API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding, or None if API call fails
        """
        if not self.use_hf_api or not self.client:
            return None
            
        if not text or not text.strip():
            return None
            
        try:
            # Use the InferenceClient to get embeddings
            result = self.client.feature_extraction(text.strip())
            
            # Handle numpy array result
            if isinstance(result, np.ndarray):
                return result.tolist()
            
            # Handle list result
            if isinstance(result, list) and len(result) > 0:
                # The API returns a list of embeddings, we want the first one
                embedding = result[0]
                if isinstance(embedding, (list, np.ndarray)):
                    if isinstance(embedding, np.ndarray):
                        return embedding.tolist()
                    return embedding
                    
            return None
            
        except Exception as e:
            print(f"⚠️  Hugging Face API request failed: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Dictionary mapping text to embedding (or None if failed)
        """
        results = {}
        for text in texts:
            results[text] = self.get_embedding(text)
        return results
    
    def setup_tfidf_fallback(self, documents: List[str]) -> None:
        """
        Setup TF-IDF fallback for when Hugging Face API is not available.
        
        Args:
            documents: List of documents to build TF-IDF index from
        """
        if not documents:
            return
            
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            self.tfidf_documents = documents
            print(f"✅ TF-IDF fallback index built with {len(documents)} documents")
        except Exception as e:
            print(f"⚠️  Failed to build TF-IDF fallback index: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.tfidf_documents = []
    
    def get_tfidf_similarity(self, query: str, document_ids: List[str]) -> Dict[str, float]:
        """
        Get TF-IDF similarity scores for a query against indexed documents.
        
        Args:
            query: Query text
            document_ids: List of document IDs corresponding to the TF-IDF matrix
            
        Returns:
            Dictionary mapping document ID to similarity score
        """
        if (self.tfidf_vectorizer is None or 
            self.tfidf_matrix is None or 
            not document_ids):
            return {}
            
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = (self.tfidf_matrix @ query_vector.T).toarray().ravel()
            
            return {
                doc_id: float(similarities[i]) 
                for i, doc_id in enumerate(document_ids)
            }
        except Exception as e:
            print(f"⚠️  TF-IDF similarity calculation failed: {e}")
            return {}
    
    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            a = np.array(vec_a, dtype=np.float32)
            b = np.array(vec_b, dtype=np.float32)
            
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            print(f"⚠️  Cosine similarity calculation failed: {e}")
            return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the embeddings helper.
        
        Returns:
            Dictionary with status information
        """
        return {
            "hf_api_available": self.use_hf_api,
            "model_name": self.model_name,
            "tfidf_fallback_ready": self.tfidf_vectorizer is not None,
            "tfidf_documents_count": len(self.tfidf_documents) if self.tfidf_documents else 0
        }


def get_embedding(text: str, api_token: Optional[str] = None) -> Optional[List[float]]:
    """
    Convenience function to get a single embedding.
    
    Args:
        text: Text to embed
        api_token: Optional Hugging Face API token
        
    Returns:
        Embedding vector or None if failed
    """
    hf_embeddings = HuggingFaceEmbeddings(api_token)
    return hf_embeddings.get_embedding(text)


def test_huggingface_api(api_token: Optional[str] = None) -> bool:
    """
    Test if the Hugging Face API is working.
    
    Args:
        api_token: Optional Hugging Face API token
        
    Returns:
        True if API is working, False otherwise
    """
    hf_embeddings = HuggingFaceEmbeddings(api_token)
    test_embedding = hf_embeddings.get_embedding("test")
    return test_embedding is not None


if __name__ == "__main__":
    # Test the embeddings helper
    print("Testing Hugging Face Embeddings Helper...")
    
    # Test with environment variable
    hf_embeddings = HuggingFaceEmbeddings()
    status = hf_embeddings.get_status()
    print(f"Status: {status}")
    
    if status["hf_api_available"]:
        print("✅ Hugging Face API is available")
        test_embedding = hf_embeddings.get_embedding("This is a test sentence.")
        if test_embedding:
            print(f"✅ Test embedding generated: {len(test_embedding)} dimensions")
        else:
            print("❌ Test embedding failed")
    else:
        print("⚠️  Hugging Face API not available (no token or token invalid)")
        print("Will use TF-IDF fallback")
