import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """A simplified in-memory vector store implementation to replace ChromaDB."""
    
    def __init__(self, persist_directory="./simple_store", embedding_model=None):
        """Initialize the SimpleVectorStore with specified parameters.
        
        Args:
            persist_directory (str): Directory to persist embeddings (not used in this simple implementation)
            embedding_model (SentenceTransformer, optional): Pre-loaded embedding model
        """
        try:
            # Create persist directory if it doesn't exist (for future persistence implementation)
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize or use provided embedding model
            self.embedding_model = embedding_model
            if self.embedding_model is None:
                logger.info("Loading default embedding model")
                self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            
            # Initialize in-memory storage
            self.documents = []
            self.embeddings = []
            self.metadatas = []
            self.ids = []
            
            logger.info("SimpleVectorStore initialized")
        except Exception as e:
            logger.error(f"Error initializing SimpleVectorStore: {str(e)}")
            raise
    
    def _generate_embedding(self, text):
        """Generate embedding for the given text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding as a numpy array
        """
        try:
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def add_documents(self, documents, paper_id):
        """Add document chunks to the vector store.
        
        Args:
            documents (list): List of document chunks
            paper_id (str): ID of the paper these chunks belong to
            
        Returns:
            int: Number of chunks added
        """
        try:
            logger.info(f"Adding {len(documents)} chunks to vector store for paper {paper_id}")
            
            # Generate IDs for each chunk
            new_ids = [f"{paper_id}_chunk_{i}" for i in range(len(documents))]
            
            # Generate embeddings for each chunk
            for i, chunk in enumerate(tqdm(documents, desc="Embedding chunks")):
                embedding = self._generate_embedding(chunk)
                metadata = {"paper_id": paper_id, "chunk_index": i}
                
                self.ids.append(new_ids[i])
                self.embeddings.append(embedding)
                self.documents.append(chunk)
                self.metadatas.append(metadata)
            
            logger.info(f"Added {len(documents)} chunks to vector store")
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, query, n_results=5):
        """Search for similar documents based on the query using cosine similarity.
        
        Args:
            query (str): Query text
            n_results (int): Number of results to return
            
        Returns:
            list: List of similar documents with metadata
        """
        try:
            logger.info(f"Searching for: {query}")
            
            if not self.embeddings:
                logger.warning("No documents in the vector store")
                return []
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Convert list of embeddings to numpy array for efficient computation
            embeddings_array = np.array(self.embeddings)
            
            # Calculate cosine similarity
            similarities = np.dot(embeddings_array, query_embedding) / (
                np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get indices of top n_results
            if len(similarities) <= n_results:
                top_indices = np.argsort(similarities)[::-1]
            else:
                top_indices = np.argsort(similarities)[::-1][:n_results]
            
            # Format results
            formatted_results = []
            for idx in top_indices:
                formatted_results.append({
                    "id": self.ids[idx],
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "distance": 1 - similarities[idx]  # Convert similarity to distance
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def get_paper_chunks(self, paper_id):
        """Get all chunks for a specific paper.
        
        Args:
            paper_id (str): ID of the paper
            
        Returns:
            list: List of chunks with metadata
        """
        try:
            logger.info(f"Retrieving chunks for paper: {paper_id}")
            
            # Filter by paper_id
            formatted_results = []
            for i, metadata in enumerate(self.metadatas):
                if metadata.get("paper_id") == paper_id:
                    formatted_results.append({
                        "id": self.ids[i],
                        "document": self.documents[i],
                        "metadata": metadata
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} chunks")
            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving paper chunks: {str(e)}")
            raise
    
    def delete_paper(self, paper_id):
        """Delete all chunks for a specific paper.
        
        Args:
            paper_id (str): ID of the paper
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Deleting chunks for paper: {paper_id}")
            
            # Find indices to delete
            indices_to_delete = []
            for i, metadata in enumerate(self.metadatas):
                if metadata.get("paper_id") == paper_id:
                    indices_to_delete.append(i)
            
            # Delete in reverse order to avoid index shifting
            for idx in sorted(indices_to_delete, reverse=True):
                del self.ids[idx]
                del self.documents[idx]
                del self.embeddings[idx]
                del self.metadatas[idx]
            
            logger.info(f"Deleted {len(indices_to_delete)} chunks for paper {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting paper chunks: {str(e)}")
            raise