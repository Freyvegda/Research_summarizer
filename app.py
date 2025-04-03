import os
import logging
import argparse
from src.query_processor import QueryProcessor
from src.paper_retriever import PaperRetriever
from src.document_processor import DocumentProcessor
from src.simple_vector_store import SimpleVectorStore
from src.summarizer import Summarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchPaperSummarizer:
    """Main class for the AI-Enhanced Research Paper Summarizer workflow."""
    
    def __init__(self, max_papers=5, chunk_size=1000, chunk_overlap=200):
        """Initialize the ResearchPaperSummarizer with specified parameters.
        
        Args:
            max_papers (int): Maximum number of papers to retrieve
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks
        """
        try:
            logger.info("Initializing ResearchPaperSummarizer")
            
            # Create necessary directories
            os.makedirs("./downloads", exist_ok=True)
            os.makedirs("./chroma_db", exist_ok=True)
            
            # Initialize components
            self.query_processor = QueryProcessor()
            self.paper_retriever = PaperRetriever(max_results=max_papers)
            self.document_processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            self.vector_store = SimpleVectorStore(
                persist_directory="./simple_store",
                embedding_model=self.query_processor.model
            )
            self.summarizer = Summarizer()
            
            logger.info("ResearchPaperSummarizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ResearchPaperSummarizer: {str(e)}")
            raise
    
    def process_query(self, query, max_papers=None, date_filter=None, categories=None):
        """Process a user query through the entire workflow.
        
        Args:
            query (str): User's research query
            max_papers (int, optional): Maximum number of papers to retrieve
            date_filter (int, optional): Filter papers published within the last n days
            categories (list, optional): List of arXiv categories to include
            
        Returns:
            dict: Dictionary containing summary and metadata
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Process the query
            processed_query = self.query_processor.process_query(query)
            logger.info(f"Preprocessed query: {processed_query['preprocessed_query']}")
            
            # Step 2: Retrieve relevant papers
            papers = self.paper_retriever.search_papers(
                query=processed_query['preprocessed_query'],
                max_results=max_papers
            )
            
            # Filter papers if needed
            if date_filter is not None or categories is not None:
                papers = self.paper_retriever.filter_papers(
                    papers=papers,
                    date_filter=date_filter,
                    categories=categories
                )
            
            logger.info(f"Retrieved {len(papers)} papers")
            
            # Step 3: Process papers and store in vector database
            paper_metadata = {}
            for paper in papers:
                # Extract metadata
                metadata = self.paper_retriever.get_paper_metadata(paper)
                paper_id = metadata['id']
                paper_metadata[paper_id] = metadata
                
                # Download paper
                pdf_path = self.paper_retriever.download_paper(paper)
                
                # Process document
                document = self.document_processor.process_document(pdf_path)
                
                # Store in vector database
                self.vector_store.add_documents(document['chunks'], paper_id)
            
            # Step 4: Search for relevant chunks based on the query
            retrieved_chunks = self.vector_store.search(
                query=processed_query['preprocessed_query'],
                n_results=10
            )
            
            # Step 5: Generate RAG summary
            summary_result = self.summarizer.generate_rag_summary(
                query=query,
                retrieved_chunks=retrieved_chunks,
                paper_metadata=paper_metadata
            )
            
            logger.info("Query processing complete")
            return summary_result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def format_summary_output(self, summary_result):
        """Format the summary result for display.
        
        Args:
            summary_result (dict): Summary result from process_query
            
        Returns:
            str: Formatted summary output
        """
        try:
            output = []
            
            # Add query
            output.append(f"Query: {summary_result['query']}\n")
            
            # Add summary
            output.append("Summary:")
            output.append(f"{summary_result['summary']}\n")
            
            # Add sources
            output.append("Sources:")
            for i, source in enumerate(summary_result['sources'], 1):
                output.append(f"{i}. {source.get('title', 'Unknown Title')}")
                output.append(f"   Authors: {', '.join(source.get('authors', ['Unknown']))}")
                output.append(f"   URL: {source.get('pdf_url', 'N/A')}")
                output.append("")
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error formatting summary output: {str(e)}")
            return str(summary_result)

def main():
    """Main function to run the ResearchPaperSummarizer."""
    parser = argparse.ArgumentParser(description="AI-Enhanced Research Paper Summarizer")
    parser.add_argument("--query", type=str, help="Research query")
    parser.add_argument("--max_papers", type=int, default=5, help="Maximum number of papers to retrieve")
    parser.add_argument("--date_filter", type=int, help="Filter papers published within the last n days")
    parser.add_argument("--categories", type=str, help="Comma-separated list of arXiv categories")
    
    args = parser.parse_args()
    
    # If no query provided, prompt user
    query = args.query
    if not query:
        query = input("Enter your research query: ")
    
    # Parse categories if provided
    categories = None
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(',')]
    
    # Initialize and run the summarizer
    summarizer = ResearchPaperSummarizer(max_papers=args.max_papers)
    summary_result = summarizer.process_query(
        query=query,
        max_papers=args.max_papers,
        date_filter=args.date_filter,
        categories=categories
    )
    
    # Format and print the result
    formatted_output = summarizer.format_summary_output(summary_result)
    print("\n" + "=" * 80 + "\n")
    print(formatted_output)
    print("=" * 80)

if __name__ == "__main__":
    main()