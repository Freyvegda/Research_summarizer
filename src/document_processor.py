import os
import logging
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class for processing PDF documents and chunking text."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Initialize the DocumentProcessor with specified parameters.
        
        Args:
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        logger.info("DocumentProcessor initialized")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary containing extracted text by page
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            # Open the PDF file
            doc = fitz.open(pdf_path)
            
            # Extract text from each page
            text_by_page = {}
            for page_num, page in enumerate(tqdm(doc, desc="Extracting pages")):
                text = page.get_text()
                text_by_page[page_num] = text
            
            # Close the document
            doc.close()
            
            logger.info(f"Extracted text from {len(text_by_page)} pages")
            return text_by_page
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_metadata_from_pdf(self, pdf_path):
        """Extract metadata from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary containing PDF metadata
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            logger.info(f"Extracting metadata from PDF: {pdf_path}")
            
            # Open the PDF file
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = doc.metadata
            
            # Add page count
            metadata['page_count'] = len(doc)
            
            # Close the document
            doc.close()
            
            logger.info(f"Extracted metadata from PDF")
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text):
        """Split text into chunks.
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            list: List of text chunks
        """
        try:
            logger.info(f"Chunking text of length {len(text)}")
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def process_document(self, pdf_path):
        """Process a PDF document by extracting text and chunking.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary containing document metadata and chunks
        """
        try:
            logger.info(f"Processing document: {pdf_path}")
            
            # Extract text and metadata
            text_by_page = self.extract_text_from_pdf(pdf_path)
            metadata = self.extract_metadata_from_pdf(pdf_path)
            
            # Combine text from all pages
            full_text = "\n".join([text for _, text in sorted(text_by_page.items())])
            
            # Chunk the text
            chunks = self.chunk_text(full_text)
            
            # Create document object
            document = {
                "metadata": metadata,
                "chunks": chunks,
                "pdf_path": pdf_path
            }
            
            logger.info(f"Document processing complete")
            return document
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Assuming a PDF file exists at this path
    # document = processor.process_document("./downloads/sample_paper.pdf")
    # print(f"Document metadata: {document['metadata']}")
    # print(f"Number of chunks: {len(document['chunks'])}")
    # print(f"First chunk: {document['chunks'][0][:200]}...")