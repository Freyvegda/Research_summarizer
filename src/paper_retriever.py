import arxiv
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperRetriever:
    """Class for retrieving relevant research papers from arXiv."""
    
    def __init__(self, max_results=10):
        """Initialize the PaperRetriever with specified parameters.
        
        Args:
            max_results (int): Maximum number of results to return
        """
        self.client = arxiv.Client()
        self.max_results = max_results
        logger.info("PaperRetriever initialized")
    
    def search_papers(self, query, max_results=None, sort_by='relevance', sort_order='descending'):
        """Search for papers on arXiv based on the query.
        
        Args:
            query (str): Search query
            max_results (int, optional): Maximum number of results to return
            sort_by (str): Sort results by 'relevance', 'lastUpdatedDate', or 'submittedDate'
            sort_order (str): Sort order, 'ascending' or 'descending'
            
        Returns:
            list: List of paper objects
        """
        try:
            if max_results is None:
                max_results = self.max_results
                
            logger.info(f"Searching arXiv for: {query}")
            
            # Map sort parameters to arxiv API parameters
            sort_by_map = {
                'relevance': arxiv.SortCriterion.Relevance,
                'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
                'submittedDate': arxiv.SortCriterion.SubmittedDate
            }
            
            sort_order_map = {
                'ascending': arxiv.SortOrder.Ascending,
                'descending': arxiv.SortOrder.Descending
            }
            
            # Create search object
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by_map.get(sort_by, arxiv.SortCriterion.Relevance),
                sort_order=sort_order_map.get(sort_order, arxiv.SortOrder.Descending)
            )
            
            # Execute search and collect results
            papers = list(tqdm(self.client.results(search), total=max_results, desc="Fetching papers"))
            logger.info(f"Retrieved {len(papers)} papers from arXiv")
            
            return papers
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            raise
    
    def filter_papers(self, papers, date_filter=None, categories=None):
        """Filter papers based on date and categories.
        
        Args:
            papers (list): List of paper objects
            date_filter (int, optional): Filter papers published within the last n days
            categories (list, optional): List of arXiv categories to include
            
        Returns:
            list: Filtered list of paper objects
        """
        try:
            filtered_papers = papers
            
            # Filter by date if specified
            if date_filter is not None:
                cutoff_date = datetime.now() - timedelta(days=date_filter)
                filtered_papers = [paper for paper in filtered_papers 
                                if paper.published > cutoff_date]
            
            # Filter by categories if specified
            if categories is not None:
                filtered_papers = [paper for paper in filtered_papers 
                                if any(category in paper.categories for category in categories)]
            
            logger.info(f"Filtered to {len(filtered_papers)} papers")
            return filtered_papers
        except Exception as e:
            logger.error(f"Error filtering papers: {str(e)}")
            return papers  # Return original papers if filtering fails
    
    def download_paper(self, paper, download_dir="./downloads"):
        """Download a paper's PDF.
        
        Args:
            paper (arxiv.Result): Paper object
            download_dir (str): Directory to save downloaded papers
            
        Returns:
            str: Path to downloaded PDF
        """
        try:
            import os
            os.makedirs(download_dir, exist_ok=True)
            
            # Generate a filename based on paper ID
            filename = f"{download_dir}/{paper.get_short_id().replace('/', '_')}.pdf"
            
            logger.info(f"Downloading paper: {paper.title}")
            paper.download_pdf(filename=filename)
            logger.info(f"Downloaded paper to {filename}")
            
            return filename
        except Exception as e:
            logger.error(f"Error downloading paper: {str(e)}")
            raise
    
    def get_paper_metadata(self, paper):
        """Extract metadata from a paper object.
        
        Args:
            paper (arxiv.Result): Paper object
            
        Returns:
            dict: Dictionary containing paper metadata
        """
        try:
            metadata = {
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "categories": paper.categories,
                "published": paper.published,
                "updated": paper.updated,
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id
            }
            return metadata
        except Exception as e:
            logger.error(f"Error extracting paper metadata: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    retriever = PaperRetriever(max_results=5)
    papers = retriever.search_papers("quantum computing")
    
    for paper in papers:
        metadata = retriever.get_paper_metadata(paper)
        print(f"Title: {metadata['title']}")
        print(f"Authors: {', '.join(metadata['authors'])}")
        print(f"Abstract: {metadata['abstract'][:200]}...")
        print("---")