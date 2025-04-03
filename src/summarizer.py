import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Summarizer:
    """Class for generating summaries using RAG approach."""
    
    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        """Initialize the Summarizer with a specified model.
        
        Args:
            model_name (str): Name of the summarization model to use
            device (str, optional): Device to run the model on ('cpu', 'cuda', etc.)
        """
        try:
            # Determine device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading summarization model: {model_name} on {device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Get model's max position embeddings (usually 1024 for BART)
            self.max_position_embeddings = self.model.config.max_position_embeddings
            if hasattr(self.model.config, 'max_position_embeddings'):
                self.max_position_embeddings = self.model.config.max_position_embeddings
            else:
                # Default for BART if not specified
                self.max_position_embeddings = 1024
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            
            logger.info(f"Summarizer initialized with max position embeddings: {self.max_position_embeddings}")
        except Exception as e:
            logger.error(f"Error initializing Summarizer: {str(e)}")
            raise
    
    def _chunk_text_for_model(self, text, max_length=None):
        """Split text into chunks that fit within model's max token limit.
        
        Args:
            text (str): Text to split
            max_length (int, optional): Maximum token length for the model
            
        Returns:
            list: List of text chunks
        """
        try:
            # Use model's max position embeddings if max_length not specified
            if max_length is None:
                # Use a slightly smaller value to account for special tokens
                max_length = self.max_position_embeddings - 100
            
            # Ensure max_length is within model's capabilities
            max_length = min(max_length, self.max_position_embeddings - 100)
            
            tokens = self.tokenizer.encode(text)
            
            # If text fits within limit, return as is
            if len(tokens) <= max_length:
                return [text]
            
            # Otherwise, split into chunks
            chunks = []
            current_chunk = []
            current_length = 0
            
            for token in tokens:
                if current_length + 1 > max_length:
                    # Convert tokens to text and add to chunks
                    chunk_text = self.tokenizer.decode(current_chunk, skip_special_tokens=True)
                    chunks.append(chunk_text)
                    current_chunk = [token]
                    current_length = 1
                else:
                    current_chunk.append(token)
                    current_length += 1
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunk_text = self.tokenizer.decode(current_chunk, skip_special_tokens=True)
                chunks.append(chunk_text)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text for model: {str(e)}")
            raise
    
    def summarize_text(self, text, max_length=500, min_length=200):
        """Generate a summary for the given text.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            str: Generated summary
        """
        try:
            logger.info(f"Summarizing text of length {len(text)}")
            
            # Split text into chunks if needed
            text_chunks = self._chunk_text_for_model(text)
            
            # Generate summary for each chunk
            summaries = []
            for chunk in tqdm(text_chunks, desc="Summarizing chunks"):
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    summaries.append(summary[0]['summary_text'])
                except Exception as chunk_error:
                    logger.warning(f"Error summarizing chunk: {str(chunk_error)}. Skipping this chunk.")
                    # Add a placeholder for failed chunks to maintain context
                    summaries.append("[Content summarization failed for this section]")
            
            # Combine summaries
            combined_summary = " ".join(summaries)
            
            # If combined summary is too long, summarize again
            if len(combined_summary.split()) > max_length and len(text_chunks) > 1:
                logger.info("Combined summary too long, summarizing again")
                combined_summary = self.summarize_text(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length
                )
            
            logger.info("Summarization complete")
            return combined_summary
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise
    
    def generate_rag_summary(self, query, retrieved_chunks, paper_metadata=None, max_length=800, min_length=300):
        """Generate a summary using RAG approach.
        
        Args:
            query (str): User query
            retrieved_chunks (list): List of retrieved text chunks
            paper_metadata (dict, optional): Metadata about the papers
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            dict: Dictionary containing summary and metadata
        """
        try:
            logger.info(f"Generating RAG summary for query: {query}")
            
            # Combine retrieved chunks
            combined_text = "\n\n".join([chunk['document'] for chunk in retrieved_chunks])
            
            # Generate summary
            summary = self.summarize_text(
                combined_text,
                max_length=max_length,
                min_length=min_length
            )
            
            # Create result object
            result = {
                "query": query,
                "summary": summary,
                "sources": []
            }
            
            # Add source information
            paper_ids = set()
            for chunk in retrieved_chunks:
                paper_id = chunk['metadata']['paper_id']
                if paper_id not in paper_ids:
                    paper_ids.add(paper_id)
                    if paper_metadata and paper_id in paper_metadata:
                        result['sources'].append(paper_metadata[paper_id])
                    else:
                        result['sources'].append({"id": paper_id})
            
            logger.info(f"Generated RAG summary with {len(result['sources'])} sources")
            return result
        except Exception as e:
            logger.error(f"Error generating RAG summary: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    summarizer = Summarizer()
    
    # Example text
    sample_text = """
    Quantum computing is an emerging field that leverages quantum mechanics to process information in ways 
    that classical computers cannot. Unlike classical bits, which can be either 0 or 1, quantum bits or qubits 
    can exist in multiple states simultaneously due to superposition. This property, along with entanglement, 
    allows quantum computers to perform certain calculations exponentially faster than classical computers.
    
    Recent advancements in quantum computing include improvements in qubit coherence times, error correction 
    methods, and the development of quantum algorithms for specific problems. Companies like IBM, Google, and 
    Microsoft are investing heavily in quantum hardware and software development.
    """
    
    summary = summarizer.summarize_text(sample_text)
    print(f"Summary: {summary}")