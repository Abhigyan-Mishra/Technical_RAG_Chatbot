# Advanced RAG Chatbot for Technical Documentation

This project implements an advanced Retrieval Augmented Generation (RAG) chatbot designed to answer questions based on a fixed set of technical documentation PDFs. It leverages local LLMs via Ollama and incorporates several techniques to improve response quality, ensure factual grounding, provide detailed references, and minimize hallucinations.

## Key Features

-   **Fixed Corpus Optimization:** Designed to work with a specific set of PDF documents that are pre-processed once.
-   **Local LLMs with Ollama:** Utilizes Ollama to run language and embedding models locally, ensuring data privacy and control.
-   **Structured Output (Pydantic):** Employs Pydantic models to compel the LLM to generate answers in a structured JSON format. This includes:
    -   A comprehensive answer.
    -   Detailed references with source filename, page number, and verbatim quotes for each statement.
    -   An LLM-assessed confidence score.
    -   Notes on any issues or missing information from the context.
-   **Query Expansion:** Uses an LLM to generate multiple variations of the user's query to improve the chances of retrieving all relevant document sections.
-   **Context Re-ranking:** Retrieves an initial set of document chunks and then uses an LLM to re-rank them for relevance before passing them to the final answer generation stage.
-   **Faithfulness Check:** An additional LLM-based step to verify if the generated answer is faithful to (i.e., fully supported by) the provided context documents.
-   **Granular Referencing:** Aims to provide specific source document names, page numbers (1-indexed), and verbatim quotes for statements made in the answer.
-   **Hallucination Reduction:** Multiple layers (strict prompting, structured output, faithfulness check) work in concert to minimize the chances of the LLM inventing information not present in the source documents.
-   **Streamlit Interface:** A user-friendly web interface for interacting with the chatbot, displaying answers, and exploring references.
-   **Modular Code:** Organized into `src` with separate files for preprocessing, the Streamlit app, Pydantic models, and utilities.

## How it Works (High-Level RAG Pipeline)

1.  **Preprocessing (Offline - `src/preprocess.py`):**
    * PDF documents from the `docs_corpus/` directory are loaded using `PyMuPDFLoader`.
    * Text is extracted and split into manageable chunks based on configured `CHUNK_SIZE` and `CHUNK_OVERLAP`.
    * Source filename and page number metadata are attached to each chunk.
    * Embeddings are generated for each chunk using a specified Ollama embedding model (e.g., `nomic-embed-text`).
    * A FAISS vector store is created from these embeddings and saved to the `vector_store/` directory, along with metadata about the processed corpus. This step is run once or whenever the source documents are updated.
2.  **Chatbot Interaction (Online - `src/rag_app.py`):**
    * The pre-built FAISS vector store and necessary Ollama LLM(s) and embedding model are loaded on startup.
    * **User Query:** The user submits a question through the Streamlit interface.
    * **Query Expansion:** The user's query is passed to an LLM to generate several diverse variations.
    * **Retrieval:** All query variants are used to retrieve an initial set of potentially relevant document chunks from the FAISS vector store (e.g., top `INITIAL_RETRIEVAL_K` chunks per variant). A similarity threshold is applied.
    * **Re-ranking:** The collected unique chunks are then re-ranked by another LLM call based on their relevance to the original user query. The top `RE_RANKED_K` chunks are selected.
    * **Answer Generation:** These highly relevant, re-ranked context chunks and the original user query are fed to the primary LLM. This LLM is instructed via a detailed prompt and Pydantic model schema to generate a structured answer. The answer must include a comprehensive response, detailed references (source, page, quote for each statement), a confidence score, and any identified issues.
    * **Faithfulness Check (Optional but Recommended):** The generated structured answer and the context chunks used to create it are passed to another LLM call. This LLM assesses if the answer is fully supported by the context and notes any discrepancies.
    * **Display:** The comprehensive answer, confidence score, faithfulness check results, and detailed, expandable references (with quotes) are displayed to the user in the Streamlit interface.

## Directory Structure

advanced_rag_chatbot/
├── .gitignore
├── README.md
├── requirements.txt
├── docs_corpus/       # Place your PDF documentation files here
│   └── placeholder.txt
├── vector_store/      # Stores the generated FAISS index and metadata
│   └── placeholder.txt
├── src/
│   ├── init.py
│   ├── preprocess.py    # Script to process PDFs and create vector store
│   ├── rag_app.py       # Main Streamlit chatbot application
│   ├── pydantic_models.py # Pydantic models for structured LLM output
│   └── utils.py         # Utility functions (e.g., loading Ollama models)
└── evaluation/          # Optional: For your golden dataset & eval scripts
    └── golden_dataset_example.json

## Prerequisites

-   **Python 3.9+**
-   **Ollama:**
    * Install Ollama from [ollama.com](https://ollama.com/).
    * Ensure the Ollama application/server is running in your environment.
    * Pull the necessary LLM and embedding models using the Ollama CLI. This project uses defaults that can be configured in `src/utils.py`:
        * **LLM for Generation/JSON tasks:** `ollama pull llama3:8b` (or your preferred model that handles JSON well, e.g., specified as `DEFAULT_OLLAMA_JSON_LLM_MODEL`).
        * **Embedding Model:** `ollama pull nomic-embed-text` (or your preferred embedding model, e.g., specified as `DEFAULT_OLLAMA_EMBEDDING_MODEL`).
        * You might use the same model for all LLM tasks or different ones (e.g., a smaller model for query expansion/re-ranking if the main generation LLM is very large).

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd advanced_rag_chatbot
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Your Documents:**
    * Create the `docs_corpus/` directory at the root of the project if it doesn't exist.
    * Place all your PDF documentation files that you want the chatbot to use into this `docs_corpus/` directory.

## Configuration

Key configuration variables are located at the top of the respective Python files or within `src/utils.py`:

-   **Ollama Model Names:** Default model names for LLMs (general and JSON-specific) and embeddings are in `src/utils.py` (e.g., `DEFAULT_OLLAMA_LLM_MODEL`, `DEFAULT_OLLAMA_EMBEDDING_MODEL`). These can be overridden by setting environment variables with the same names.
-   **File Paths:** Paths for `DOCS_CORPUS_DIR`, `VECTOR_STORE_DIR`, etc., are defined in `src/preprocess.py` and `src/rag_app.py`, typically relative to the project root.
-   **Chunking Parameters (`src/preprocess.py`):**
    * `CHUNK_SIZE`: Target size of text chunks.
    * `CHUNK_OVERLAP`: Overlap between chunks.
    * **These are critical for RAG performance and should be tuned based on your document structure.**
-   **Retrieval Parameters (`src/rag_app.py`):**
    * `INITIAL_RETRIEVAL_K`: Number of chunks initially retrieved per query variant.
    * `RE_RANKED_K`: Number of chunks kept after LLM-based re-ranking for the final context.
    * `SIMILARITY_SCORE_THRESHOLD`: Threshold for initial document filtering based on vector similarity score.

Review and adjust these settings as needed for your specific documents and hardware capabilities.

## Usage

**Step 1: Preprocess Documents (Run Once, or when documents change)**

This step is crucial. It loads your PDFs from `docs_corpus/`, processes them into chunks, generates embeddings, and saves the FAISS vector store to `vector_store/`.

* **Ensure Ollama is running** with the required embedding model pulled (e.g., `ollama pull nomic-embed-text`).
* Navigate to the `src` directory (or run from the project root, ensuring Python can find the `src` modules).
    ```bash
    # From the project root:
    python src/preprocess.py
    # Or, if you are in the src directory:
    # python preprocess.py
    ```
    This will create the vector store index files (e.g., `docs_faiss_index.faiss`, `docs_faiss_index.pkl`) and a metadata file in the `vector_store/` directory.

**Step 2: Run the Chatbot Application**

Once preprocessing is complete and the vector store exists:

* **Ensure Ollama is running** with the required LLM(s) pulled (e.g., `ollama pull llama3:8b`).
* Navigate to the `src` directory (or run from the project root).
    ```bash
    # From the project root:
    streamlit run src/rag_app.py
    # Or, if you are in the src directory:
    # streamlit run rag_app.py
    ```
    This will start the Streamlit web server and open the chatbot interface in your default web browser.

## Achieving High-Quality Results & Hyperparameter Tuning

Optimizing a RAG system for specific technical documentation requires iterative tuning. This project provides a strong foundation, but consider the following for further improvements:

1.  **Chunking Strategy (`src/preprocess.py`):** This is paramount.
    * Experiment with different `CHUNK_SIZE` and `CHUNK_OVERLAP` values. Technical documents with many tables, code blocks, or figures might benefit from different strategies than dense prose. Re-run `preprocess.py` after changes.
2.  **Retrieval Parameters (`src/rag_app.py`):**
    * `INITIAL_RETRIEVAL_K`, `RE_RANKED_K`, and `SIMILARITY_SCORE_THRESHOLD` directly influence the context quality. Test different combinations.
3.  **Embedding Model (configured in `src/utils.py`):**
    * While `nomic-embed-text` is a good default, specialized documentation might benefit from other embedding models. This requires re-preprocessing.
4.  **LLM Models (configured in `src/utils.py`):**
    * The choice of LLM for generation, query expansion, re-ranking, and faithfulness checks impacts quality and speed. Larger models might offer better reasoning and JSON adherence but will be slower. Ensure your chosen models work well with the `format="json"` Ollama parameter if used for structured output.
5.  **Prompt Engineering (`src/rag_app.py`):**
    * The prompts, especially for `generate_structured_answer` and `check_answer_faithfulness`, are critical. Refine them to better suit the style, complexity, and expected output for your specific documentation. For example, if your documents often list procedural steps, instruct the LLM to preserve or summarize such formats.
6.  **"Golden Dataset" Evaluation (Manual/Semi-Automated):**
    * Create a representative set of questions with known answers and source locations from your documents (see `evaluation/golden_dataset_example.json`).
    * Systematically test your RAG pipeline against this dataset after each significant parameter change.
    * Evaluate:
        * **Context Relevance:** Are the re-ranked chunks accurate and sufficient?
        * **Answer Faithfulness:** Does the answer strictly adhere to the provided context (verified by the faithfulness check and your own review)?
        * **Reference Accuracy:** Are the cited sources (filename, page, quote) correct?
        * **Completeness & Conciseness:** Is the answer thorough for the query but not overly verbose?

## Troubleshooting

-   **Ollama Issues:**
    * Ensure the Ollama application is running or `ollama serve` is active.
    * Verify models are pulled: `ollama list`. Pull if missing: `ollama pull <model_name>`.
    * Check Ollama server logs for errors if models fail to load.
-   **Vector Store Not Found:** Confirm `src/preprocess.py` ran successfully and created files in `vector_store/`. Check paths in scripts.
-   **JSON Parsing Errors:**
    * LLMs can sometimes produce slightly malformed JSON. The `cleanup_json_string` utility in `src/utils.py` attempts to fix common issues. The `OutputFixingParser` in LangChain also tries to self-correct.
    * If errors persist:
        * Try a different LLM known for better JSON adherence (e.g., newer versions of `llama3` or models specifically tuned for function calling/JSON).
        * Simplify the Pydantic models or the prompt instructions for the JSON structure.
        * Add print statements before parsing in `src/rag_app.py` to inspect the raw/cleaned LLM output.
-   **Slow Responses:**
    * Large LLMs are inherently slower.
    * The multi-step pipeline (expansion, retrieval, re-ranking, generation, faithfulness) involves multiple LLM calls, adding to latency.
    * Ensure your hardware (CPU, RAM, GPU if used by Ollama) is adequate for the chosen models.
    * Consider using smaller/faster LLMs for auxiliary tasks like query expansion or re-ranking if the main generation LLM is large.

## Future Enhancements

-   **More Sophisticated Re-ranking:** Explore dedicated cross-encoder models for re-ranking (e.g., from Sentence Transformers library) if higher accuracy is needed and performance allows.
-   **Parent Document Retriever (LangChain):** Useful if small, precise chunks are best for initial retrieval, but larger surrounding context is needed for the LLM to generate a comprehensive answer.
-   **Automated Evaluation Pipeline:** Integrate tools like RAGAs, DeepEval, or custom scripts with your golden dataset for more quantitative and repeatable metrics on RAG performance.
-   **Knowledge Graph Integration:** For highly structured technical information (e.g., component relationships, API schemas), combining RAG with a knowledge graph could yield even more precise and interconnected answers.
-   **UI for Parameter Tuning:** Expose some key parameters (like `INITIAL_RETRIEVAL_K`, `RE_RANKED_K`, model selection) in the Streamlit UI for easier on-the-fly experimentation by advanced users.
-   **Streaming Responses:** For a better user experience with slower LLMs, implement response streaming in the Streamlit app.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
