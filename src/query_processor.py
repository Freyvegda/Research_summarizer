import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryProcessor:
    """Class for preprocessing user queries and generating embeddings."""
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """Initialize the QueryProcessor with a specified embedding model.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        try:
            # Download NLTK resources if not already present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                nltk.download('punkt')
            
            self.stop_words = set(stopwords.words('english'))
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing QueryProcessor: {str(e)}")
            raise
    
    def preprocess_query(self, query):
        """Preprocess the query by removing special characters, stopwords, etc.
        
        Args:
            query (str): The user's input query
            
        Returns:
            str: Preprocessed query
        """
        try:
            # Convert to lowercase
            query = query.lower()
            
            # Remove special characters
            query = re.sub(r'[^\w\s]', '', query)
            
            # Tokenize
            tokens = nltk.word_tokenize(query)
            
            # Remove stopwords
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            
            # Join tokens back into a string
            preprocessed_query = ' '.join(filtered_tokens)
            
            return preprocessed_query
        except Exception as e:
            logger.error(f"Error preprocessing query: {str(e)}")
            return query  # Return original query if preprocessing fails
    
    def generate_embedding(self, text):
        """Generate embedding for the given text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            torch.Tensor: Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def process_query(self, query):
        """Process the query by preprocessing and generating embedding.
        
        Args:
            query (str): The user's input query
            
        Returns:
            dict: Dictionary containing original query, preprocessed query, and embedding
        """
        try:
            logger.info(f"Processing query: {query}")
            preprocessed_query = self.preprocess_query(query)
            embedding = self.generate_embedding(preprocessed_query)
            
            return {
                "original_query": query,
                "preprocessed_query": preprocessed_query,
                "embedding": embedding
            }
        except Exception as e:
            logger.error(f"Error in query processing pipeline: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    processor = QueryProcessor()
    result = processor.process_query("What are the latest advancements in quantum computing?")
    print(f"Original Query: {result['original_query']}")
    print(f"Preprocessed Query: {result['preprocessed_query']}")
    print(f"Embedding Shape: {result['embedding'].shape}")