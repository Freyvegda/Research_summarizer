# AI-Enhanced Research Paper Summarizer

This project implements a workflow for retrieving, processing, and summarizing research papers based on user queries. It uses semantic search, the arXiv API, and retrieval-augmented generation (RAG) to provide comprehensive research summaries.

## Features

- **Query Processing**: Preprocess user queries and generate semantic embeddings
- **Research Paper Retrieval**: Fetch relevant papers from arXiv based on user queries
- **Document Processing**: Parse PDFs and chunk text for efficient processing
- **Vector Storage**: Store embeddings in a vector database for quick retrieval
- **RAG-based Summarization**: Generate comprehensive summaries using retrieved content
- **Result Presentation**: Display structured summaries with links to original papers

## Project Structure

```
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── src/
│   ├── query_processor.py  # Query preprocessing and embedding
│   ├── paper_retriever.py  # arXiv API integration
│   ├── document_processor.py # PDF parsing and text chunking
│   ├── vector_store.py     # Vector database operations
│   ├── summarizer.py       # RAG-based summarization
│   └── utils.py            # Utility functions
└── README.md               # Project documentation
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Workflow

1. **User Input & Query Processing**
   - User enters a research topic query
   - Query is preprocessed and embedded using sentence-transformers

2. **Retrieval of Relevant Research Papers**
   - Query is sent to arXiv API to fetch relevant papers
   - Results are filtered based on relevance and recency

3. **Document Processing & Chunking**
   - Papers are downloaded and converted to structured format
   - Text is split into manageable chunks

4. **Embedding & Storage**
   - Chunks are embedded and stored in a vector database

5. **RAG for Summarization**
   - Relevant chunks are retrieved based on user query
   - Open-source summarization model generates comprehensive summary

6. **Presentation of Results**
   - Summary is displayed in structured format
   - Links to original papers are provided

## Future Enhancements

- Multimodal summarization (extracting insights from figures/tables)
- Keyword extraction and topic modeling
- Citation generator for referenced papers