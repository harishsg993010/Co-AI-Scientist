import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import faiss
import pickle
from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTOR_DB_PATH, OPENAI_API_KEY
from models import Source

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI package is required for embeddings. Install with: pip install openai")

# Verify API key is available
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. Embeddings functionality will be unavailable.")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("Successfully initialized OpenAI client")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    client = None


class KnowledgeBase:
    """Vector database implementation for storing and retrieving research information"""
    
    def __init__(self, vector_db_path: str = VECTOR_DB_PATH):
        """Initialize the knowledge base"""
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.vector_db_path / "faiss_index.idx"
        self.data_path = self.vector_db_path / "document_data.pkl"
        
        # Create or load the FAISS index
        self.dimension = 1536  # OpenAI embedding dimension
        if self.index_path.exists() and self.data_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.data_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            self._save()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        if not client:
            logger.error("OpenAI client is not initialized. Cannot get embeddings.")
            # Return a zero vector of the correct dimension as a fallback
            return [0.0] * self.dimension
            
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding from OpenAI API: {str(e)}")
            # Return a zero vector of the correct dimension as a fallback
            return [0.0] * self.dimension
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into smaller chunks for embedding"""
        if len(text) <= CHUNK_SIZE:
            return [text]
        
        chunks = []
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _save(self):
        """Save the index and documents to disk"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def add_source(self, source: Source) -> bool:
        """Add a source to the knowledge base"""
        if not source or not source.content:
            logger.warning("Cannot add empty source to knowledge base")
            return False
            
        try:
            logger.info(f"Adding source to knowledge base: {source.url}")
            
            # Chunk the content
            chunks = self._chunk_text(source.content)
            logger.info(f"Source split into {len(chunks)} chunks")
            
            # Get embeddings and add to index
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                embedding = self._get_embedding(chunk)
                
                # Convert to numpy array and reshape for FAISS
                embedding_np = np.array(embedding).astype('float32').reshape(1, -1)
                
                # Add to FAISS index
                self.index.add(embedding_np)
                
                # Store document data
                doc_id = len(self.documents)
                self.documents.append({
                    'id': doc_id,
                    'chunk': chunk,
                    'source_url': source.url,
                    'source_title': source.title,
                    'source_type': source.type,
                    'added_at': datetime.now().isoformat(),
                    'metadata': source.metadata
                })
            
            # Save changes
            self._save()
            logger.info(f"Successfully added source to knowledge base: {source.url}")
            return True
        except Exception as e:
            logger.error(f"Error adding source to knowledge base: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information"""
        if not query:
            logger.warning("Empty query provided to knowledge base search")
            return []
            
        if not self.documents:
            logger.warning("Knowledge base is empty, no documents to search")
            return []
            
        try:
            logger.info(f"Searching knowledge base for: {query[:50]}...")
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)
            
            # Search the index
            distances, indices = self.index.search(query_embedding_np, top_k)
            
            # Gather results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    result = self.documents[idx].copy()
                    result['relevance_score'] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to relevance
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def get_all_sources(self) -> List[Dict[str, Any]]:
        """Get all unique sources in the knowledge base"""
        sources = {}
        for doc in self.documents:
            url = doc['source_url']
            if url not in sources:
                sources[url] = {
                    'url': url,
                    'title': doc['source_title'],
                    'type': doc['source_type'],
                    'added_at': doc['added_at'],
                    'metadata': doc.get('metadata', {})
                }
        
        return list(sources.values())
