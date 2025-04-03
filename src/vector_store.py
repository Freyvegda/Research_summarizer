import os
import logging
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """Class for managing vector embeddings in a ChromaDB database."""
    
    def __init__(self, persist_directory="./chroma_db", embedding_model=None):
        """Initialize the VectorStore with specified parameters.
        
        Args:
            persist_directory (str): Directory to persist the ChromaDB database
            embedding_model (SentenceTransformer, optional): Pre-loaded embedding model
        """
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))
            
            # Initialize or use provided embedding model
            self.embedding_model = embedding_model
            if self.embedding_model is None:
                logger.info("Loading default embedding model")
                self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            
            # Create or get collection for papers
            self.collection = self.client.get_or_create_collection(
                name="research_papers",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("VectorStore initialized")
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {str(e)}")
            raise
    
    def _generate_embedding(self, text):
        """Generate embedding for the given text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding as a list of floats
        """
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
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
            ids = [f"{paper_id}_chunk_{i}" for i in range(len(documents))]
            
            # Generate embeddings for each chunk
            embeddings = []
            metadatas = []
            
            for i, chunk in enumerate(tqdm(documents, desc="Embedding chunks")):
                embedding = self._generate_embedding(chunk)
                embeddings.append(embedding)
                metadatas.append({"paper_id": paper_id, "chunk_index": i})
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            # Persist changes
            self.client.persist()
            
            logger.info(f"Added {len(documents)} chunks to vector store")
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, query, n_results=5):
        """Search for similar documents based on the query.
        
        Args:
            query (str): Query text
            n_results (int): Number of results to return
            
        Returns:
            list: List of similar documents with metadata
        """
        try:
            logger.info(f"Searching for: {query}")
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
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
            
            # Search by metadata
            results = self.collection.get(
                where={"paper_id": paper_id}
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'])):
                formatted_results.append({
                    "id": results['ids'][i],
                    "document": results['documents'][i],
                    "metadata": results['metadatas'][i]
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
            
            # Delete by metadata
            self.collection.delete(
                where={"paper_id": paper_id}
            )
            
            # Persist changes
            self.client.persist()
            
            logger.info(f"Deleted chunks for paper {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting paper chunks: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    vector_store = VectorStore()
    
    # Example: Add documents
    # chunks = ["This is a sample chunk about quantum computing.", 
    #           "Another chunk about quantum algorithms."]
    # vector_store.add_documents(chunks, "sample_paper_001")
    
    # Example: Search
    # results = vector_store.search("quantum computing")
    # for result in results:
    #     print(f"Document: {result['document'][:100]}...")
    #     print(f"Paper ID: {result['metadata']['paper_id']}")
    #     print(f"Distance: {result['distance']}")
    #     print("---")