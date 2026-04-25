# Academic City RAG Chatbot

A Streamlit-based **Retrieval-Augmented Generation (RAG)** chatbot for Academic City that combines dense vector search with sparse keyword-based retrieval using **Reciprocal Rank Fusion (RRF)** for optimal hybrid search results.

**Author:** Lloyd Onny (ID: 10211100341)

## 🎯 Features

- **Hybrid Search**: Combines dense embeddings (FAISS + SentenceTransformers) and sparse TF-IDF keyword search
- **Reciprocal Rank Fusion**: Intelligently fuses multiple ranking strategies for better retrieval quality
- **Query Expansion**: Automatically expands queries to capture multiple search variants
- **Conversation Memory**: Optional chat history to maintain context across multiple queries
- **Prompt Templates**: Multiple response styles for customized LLM outputs
- **Pipeline Logging**: Comprehensive logging of retrieval and processing stages
- **Evaluation Baselines**: Built-in evaluation utilities for measuring retrieval quality
- **OpenAI Integration**: Uses OpenAI's API for generating intelligent responses

## 📋 Project Structure

```
ai_10211100341/
├── src/
│   ├── app.py                    # Main Streamlit application
│   ├── retrieval.py              # Hybrid RAG retriever (FAISS + TF-IDF)
│   ├── hybrid_fusion.py           # Reciprocal Rank Fusion implementation
│   ├── query_expansion.py         # Query variant generation
│   ├── llm_utils.py               # OpenAI integration & prompt templates
│   ├── embedding_index.py         # FAISS index creation
│   ├── pdf_chunking.py            # PDF document chunking
│   ├── data_cleaning.py           # Data preprocessing utilities
│   ├── evaluation_utils.py        # Evaluation metrics and baselines
│   ├── pipeline_log.py            # Logging configuration
│   └── download_data.py           # Data download utilities
├── data/
│   ├── Ghana_Election_Result.csv  # Election results dataset
│   ├── Ghana_Election_Result_clean.csv  # Cleaned election data
│   ├── 2025-Budget-Statement-and-Economic-Policy_v4.pdf  # Budget document
│   ├── budget_chunks.txt          # Chunked budget text
│   ├── chunk_metadata.csv         # Metadata for text chunks
│   ├── faiss_index.bin            # FAISS dense vector index
│   └── tfidf_keyword_index.joblib # TF-IDF sparse index
├── reports/
│   └── chunking_compare_summary.txt  # Analysis of chunking strategies
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or enter it in the Streamlit UI sidebar.

### 3. Run the Application

```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

## 📦 Dependencies

- **streamlit**: Web UI framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: TF-IDF vectorization
- **sentence-transformers**: Dense embedding model (all-MiniLM-L6-v2)
- **faiss-cpu**: Efficient similarity search
- **PyPDF2**: PDF text extraction
- **matplotlib**: Data visualization
- **openai**: LLM API integration
- **requests**: HTTP requests
- **joblib**: Index serialization

## 🔍 How It Works

### Hybrid Search Pipeline

1. **Query Expansion**: Input query is expanded into multiple variants
2. **Dense Retrieval**: Query embeddings are searched in FAISS index
3. **Sparse Retrieval**: Query is matched against TF-IDF keyword index
4. **Rank Fusion**: Dense and sparse rankings are combined using Reciprocal Rank Fusion
5. **Context Generation**: Top-k chunks are formatted as context
6. **LLM Answering**: Context is passed to OpenAI for answer generation

### Key Components

#### RAGRetriever (`retrieval.py`)
- Manages FAISS dense index and TF-IDF sparse index
- Implements hybrid search with configurable top-k results
- Supports query expansion for multi-variant search

#### Hybrid Fusion (`hybrid_fusion.py`)
- Implements Reciprocal Rank Fusion (RRF) algorithm
- Formula: Score(chunk) = Σ 1 / (K + rank)
- Combines dense and sparse ranking strategies

#### Query Expansion (`query_expansion.py`)
- Generates query variants automatically
- Captures different phrasings of the same question
- Improves recall for hybrid search

#### LLM Integration (`llm_utils.py`)
- Multiple prompt templates for different response styles
- OpenAI API wrapper with error handling
- Configurable temperature and max tokens

## 📊 Using the Application

### Sidebar Options

- **OpenAI API Key**: Enter your API key (required for responses)
- **Response Style**: Choose from multiple prompt templates
- **Remember Conversation**: Enable/disable conversation memory

### Main Interface

1. Enter your question in the text box
2. Click "Submit Query" or press Enter
3. View retrieval metrics in the expandable sections
4. Read the AI-generated response

### Evaluation Tools

The app includes built-in evaluation features to assess retrieval quality:
- Compare different chunking strategies
- Evaluate retrieval coverage and precision
- Analyze pipeline performance

## 🗂️ Data

### Datasets

- **Ghana Election Results**: Political voting data
- **Budget Statement**: 2025 Ghana budget and economic policy document

### Indexing

Run the following to rebuild indices:

```bash
# Create FAISS and TF-IDF indices
python src/embedding_index.py

# Clean and prepare data
python src/data_cleaning.py
```

## 📝 Logging & Debugging

Pipeline operations are logged automatically. Check `pipeline_log.py` for:
- Query processing stages
- Retrieval metrics
- LLM call details

Logs are helpful for debugging and understanding the retrieval pipeline.

## 🔧 Configuration

Key constants (in `retrieval.py`):

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Dense embedding model
RRF_K = 60                               # RRF parameter
RANK_POOL = 200                          # Candidates per search
```

## 📈 Performance

- **Dense Search**: ~10-50ms per query (depends on FAISS index size)
- **Sparse Search**: ~5-20ms per query
- **Fusion + Re-ranking**: <5ms
- **LLM Response**: 1-5 seconds (depends on OpenAI API)

## 🎓 Learning Resources

- **FAISS**: [Facebook AI Similarity Search](https://faiss.ai/)
- **Sentence Transformers**: [Hugging Face Models](https://www.sbert.net/)
- **Reciprocal Rank Fusion**: [Academic Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksearch.html)
- **RAG Pattern**: [LlamaIndex Guide](https://www.llamaindex.ai/)

## 🛠️ Troubleshooting

### FAISS Index Not Found

```
FileNotFoundError: Missing FAISS index at data/faiss_index.bin
```

**Solution**: Run `python src/embedding_index.py` to build indices.

### No Responses from LLM

- Verify OpenAI API key is set correctly
- Check API rate limits and account balance
- Review error messages in console

### Slow Queries

- Reduce `RANK_POOL` for faster retrieval
- Use `k=3` instead of `k=5` for fewer results
- Check system resources

## 📚 Future Improvements

- [ ] Web UI enhancements (streaming responses, pagination)
- [ ] Support for multiple LLM backends (Anthropic, Ollama)
- [ ] Advanced chunking strategies (semantic, hierarchical)
- [ ] Persistent chat history storage
- [ ] Fine-tuning embedding model
- [ ] Real-time index updates

## 📄 License

This project is created as part of academic coursework.

## 👤 Author

**Lloyd Onny** - ID: 10211100341

For questions or contributions, please open an issue or submit a pull request.
