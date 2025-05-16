# Advanced RAG Chatbot for Technical Documentation (Optimized & Debuggable)

This project implements an advanced Retrieval Augmented Generation (RAG) chatbot specifically designed to answer questions based on a **fixed set of technical PDF documents**. It leverages local Large Language Models (LLMs) via Ollama and incorporates a multi-step pipeline to enhance response quality, ensure factual grounding, provide detailed references, and minimize hallucinations. This version includes enhanced debugging features and more robust JSON parsing.

## Key Features

-   **Fixed Corpus Optimization:** Designed to work with a specific set of PDF documents that are pre-processed once for efficiency.
-   **Local LLMs with Ollama:** Utilizes Ollama to run language and embedding models locally, ensuring data privacy and control. Uses the `langchain-ollama` integration.
-   **Structured Output (Pydantic):** Employs Pydantic models to compel the LLM to generate answers in a structured JSON format. This includes:
    -   A comprehensive answer based *only* on provided context.
    -   Detailed references with source filename, page number (1-indexed), and verbatim quotes for each statement.
    -   An LLM-assessed confidence score regarding the answer's support by the context.
    -   Notes on any issues or missing information from the context.
-   **Advanced RAG Pipeline:**
    -   **Query Expansion:** Uses an LLM to generate multiple variations of the user's query to improve retrieval recall.
    -   **Context Re-ranking:** Retrieves an initial set of document chunks and then uses an LLM to re-rank them for relevance before passing them to the final answer generation stage.
    -   **Faithfulness Check:** An LLM-based step to verify if the generated answer is faithful to (i.e., fully supported by) the provided context documents.
-   **Granular Referencing:** Aims to provide specific source document names, page numbers, and verbatim quotes for statements made in the answer.
-   **Hallucination Reduction:** Multiple layers (strict prompting, structured output, faithfulness check) work in concert to minimize the chances of the LLM inventing information not present in the source documents.
-   **Streamlit Interface:** A user-friendly web interface for interacting with the chatbot, displaying answers, and exploring references. Includes:
    -   Configurable retrieval parameters (Similarity Threshold, Initial K, Re-ranked K) in the sidebar for live tuning.
    -   A persistent debug expander showing detailed intermediate steps of the retrieval and re-ranking process for the last query.
-   **Modular Code:** Organized into an `src` directory with separate files for preprocessing, the Streamlit app, Pydantic models, and utility functions.
-   **Robust JSON Handling:** Includes utilities for cleaning LLM-generated JSON and uses LangChain's `OutputFixingParser` as a fallback.

## How it Works (High-Level RAG Pipeline)

1.  **Preprocessing (Offline - `src/preprocess.py`):**
    * PDF documents from the `docs_corpus/` directory are loaded using `PyMuPDFLoader`.
    * Text is extracted and split into manageable chunks based on configured `CHUNK_SIZE` and `CHUNK_OVERLAP`.
    * Source filename and page number metadata are attached to each chunk.
    * Embeddings are generated for each chunk using a specified Ollama embedding model (e.g., `nomic-embed-text`) via `langchain-ollama`.
    * A FAISS vector store is created from these embeddings and saved to the `vector_store/` directory, along with metadata about the processed corpus. This step is run once or whenever the source documents are updated.
2.  **Chatbot Interaction (Online - `src/rag_app.py`):**
    * The pre-built FAISS vector store and necessary Ollama LLM(s) (using `OllamaLLM` from `langchain-ollama`) and embedding model are loaded on startup. Caching is used for these resources within the Streamlit app.
    * **User Query:** The user submits a question through the Streamlit interface.
    * **Query Expansion:** The user's query is passed to an LLM to generate several diverse variations.
    * **Retrieval:** All query variants are used to retrieve an initial set of potentially relevant document chunks from the FAISS vector store (e.g., top `initial_k_ui` chunks per variant). A user-configurable `similarity_threshold_ui` (L2 distance) is applied.
    * **Re-ranking:** The collected unique chunks that pass the threshold are then re-ranked by another LLM call based on their relevance to the original user query. The top `rerank_k_ui` chunks are selected.
    * **Answer Generation:** These highly relevant, re-ranked context chunks and the original user query are fed to the primary LLM. This LLM is instructed via a detailed prompt and Pydantic model schema to generate a structured answer. The answer must include a comprehensive response, detailed references (source, page, quote for each statement), a confidence score, and any identified issues, based *only* on the provided context.
    * **Faithfulness Check (Optional but Recommended):** The generated structured answer and the context chunks used to create it are passed to another LLM call. This LLM assesses if the answer is fully supported by the context and notes any discrepancies.
    * **Display:** The comprehensive answer, confidence score, faithfulness check results, and detailed, expandable references (with quotes) are displayed to the user. Debug information from the retrieval and re-ranking steps for the last query is shown in a separate expander.

## Directory Structure
advanced_rag_chatbot/
├── .gitignore                 # Specifies intentionally untracked files that Git should ignore
├── LICENSE                    # Project license file (e.g., MIT)
├── README.md                  # This file
├── requirements.txt           # Python dependencies for the project
├── docs_corpus/               # ACTION REQUIRED: Place your PDF documentation files here
├── vector_store/              # Stores the generated FAISS index and metadata (created by preprocess.py)
├── src/                       # Source code for the RAG application
│   ├── init.py            # Makes 'src' a Python package
│   ├── preprocess.py        # Script to process PDFs and create the vector store
│   ├── rag_app.py           # Main Streamlit chatbot application
│   ├── pydantic_models.py   # Pydantic models for structured LLM output and validation
│   └── utils.py             # Utility functions (e.g., loading Ollama models, JSON cleanup)
└── evaluation/                # Optional: For your golden dataset & custom evaluation scripts
└── golden_dataset_example.json # Example format for a golden dataset

## Prerequisites

-   **Python 3.9+**
-   **Ollama:**
    * Install Ollama from [ollama.com](https://ollama.com/).
    * Ensure the Ollama application/server is running in your environment.
    * Pull the necessary LLM and embedding models using the Ollama CLI. This project uses defaults that can be configured in `src/utils.py`:
        * **LLM for Generation/JSON tasks:** `ollama pull llama3:8b` (or your preferred model that handles JSON well, e.g., specified as `DEFAULT_OLLAMA_JSON_LLM_MODEL`).
        * **Embedding Model:** `ollama pull nomic-embed-text` (or your preferred embedding model, e.g., specified as `DEFAULT_OLLAMA_EMBEDDING_MODEL`).
        * You might use the same model for all LLM tasks or different ones. The LLMs used for JSON generation should support the `format="json"` parameter in Ollama.

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
    *(Ensure `requirements.txt` includes `langchain-ollama>=0.1.0` and `langchain-community` for FAISS).*

4.  **Prepare Your Documents:**
    * Create the `docs_corpus` directory at the root of the project if it doesn't exist.
    * Place all your PDF documentation files that you want the chatbot to use into this `docs_corpus/` directory.

## Configuration

Key configuration variables are located at the top of the respective Python files or within `src/utils.py`:

-   **Ollama Model Names:** Default model names for LLMs (general and JSON-specific) and embeddings are in `src/utils.py` (e.g., `DEFAULT_OLLAMA_LLM_MODEL`, `DEFAULT_OLLAMA_EMBEDDING_MODEL`). These can be overridden by setting environment variables with the same names.
-   **File Paths:** Paths for `DOCS_CORPUS_DIR`, `VECTOR_STORE_DIR`, etc., are defined in `src/preprocess.py` and `src/rag_app.py`, typically relative to the project root.
-   **Chunking Parameters (`src/preprocess.py`):**
    * `CHUNK_SIZE`: Target size of text chunks.
    * `CHUNK_OVERLAP`: Overlap between chunks.
    * **These are critical for RAG performance and should be tuned based on your document structure and content.**
-   **Retrieval Parameters (Defaults in `src/rag_app.py`, tunable in UI):**
    * `INITIAL_RETRIEVAL_K_DEFAULT`: Number of chunks initially retrieved per query variant.
    * `RE_RANKED_K_DEFAULT`: Number of chunks kept after LLM-based re-ranking for the final context.
    * `SIMILARITY_SCORE_THRESHOLD_DEFAULT`: Initial threshold for document filtering based on vector similarity score (L2 distance).

Review and adjust these settings as needed for your specific documents and hardware capabilities.

## Usage

**Step 1: Preprocess Documents (Run Once, or when documents change)**

This step loads your PDFs from `docs_corpus/`, processes them into chunks, generates embeddings, and saves the FAISS vector store to `vector_store/`.

* **Ensure Ollama is running** with the required embedding model pulled (e.g., `ollama pull nomic-embed-text`).
* Navigate to the project root directory and run:
    ```bash
    python src/preprocess.py
    ```
    This will create the vector store index files (e.g., `docs_faiss_index.faiss`, `docs_faiss_index.pkl`) and a metadata file in the `vector_store/` directory. Monitor the console output for success or errors.

**Step 2: Run the Chatbot Application**

Once preprocessing is complete and the vector store exists:

* **Ensure Ollama is running** with the required LLM(s) pulled (e.g., `ollama pull llama3:8b`).
* Navigate to the project root directory and run:
    ```bash
    streamlit run src/rag_app.py
    ```
    This will start the Streamlit web server and open the chatbot interface in your default web browser.

## Achieving High-Quality Results & Hyperparameter Tuning

Optimizing a RAG system for specific technical documentation requires iterative tuning.

**Key Areas for Tuning (using the Streamlit UI and by modifying `preprocess.py`):**

1.  **Retrieval Parameters (Live in Streamlit UI):**
    * **Similarity Score Threshold:** This is crucial. For L2 distance (used by FAISS default), *smaller scores are more similar*. If no documents are retrieved, your scores might be too high. Start by setting this slider to a higher value (e.g., 2.5, 3.0, or even 3.5) to see the raw scores of initially retrieved documents in the "Retrieval & Re-ranking Debug Info" expander. Then, adjust it downwards to filter out less relevant chunks.
    * `Initial Chunks to Retrieve`: How many documents to fetch initially per query variant.
    * `Final Chunks to Use`: How many documents to pass to the LLM after re-ranking.

2.  **Chunking Strategy (`src/preprocess.py`):**
    * The `CHUNK_SIZE` and `CHUNK_OVERLAP` significantly impact the quality of retrieved context. Technical documents with many tables, code blocks, or figures might benefit from different strategies than dense prose.
    * **Experiment:** Try different values, **re-run `python src/preprocess.py` each time**, and then test with a representative set of questions using the Streamlit app. Observe the debug info.

3.  **Embedding Model (configured in `src/utils.py`):**
    * If retrieval scores are consistently very poor (e.g., L2 distances are always very high even for queries you know should match), the default embedding model (`nomic-embed-text`) might not be optimal for your specific document's content (e.g., highly symbolic, mathematical, or niche jargon).
    * Consider trying other models available via Ollama (e.g., `mxbai-embed-large`). This requires changing the model name in `utils.py` and **re-running `preprocess.py`**.

4.  **LLM Models (configured in `src/utils.py`):**
    * The choice of LLM for generation, query expansion, re-ranking, and faithfulness checks impacts quality and speed. Ensure your chosen models work well with the `format="json"` Ollama parameter if used for structured output. `llama3:8b` is a good starting point.

5.  **Prompt Engineering (`src/rag_app.py`):**
    * The prompts (especially for `generate_structured_answer` and `check_answer_faithfulness`) are critical. They have been designed to be robust, but you can tailor them further to the style and content of your specific documentation.

**"Golden Dataset" Evaluation Strategy:**

1.  **Create a "Golden Dataset":** In `evaluation/`, create a JSON file (e.g., `golden_dataset.json` based on the example) with diverse questions, expected answer summaries, and ideal source locations from your documents.
2.  **Iterate and Test:** Modify one set of parameters, re-preprocess if needed, and manually run your golden dataset questions through the `rag_app.py`.
3.  **Compare and Analyze:** Check the chatbot's answers, references, confidence, and faithfulness against your golden set. Use the debug expander to understand retrieval performance.

## Troubleshooting

-   **`ImportError: cannot import name 'OllamaLLM' from 'langchain_ollama'` (or similar for `OllamaEmbeddings`):**
    * Ensure you have `langchain-ollama>=0.1.0` in your `requirements.txt` and it's correctly installed in your **active virtual environment**.
    * **Crucially, check for name shadowing:** Make sure you don't have a local file named `langchain_ollama.py` or a folder named `langchain_ollama` in your project's `src` directory or root directory. Python might be trying to import from your local file instead of the installed package.
    * Perform a clean reinstall: `pip uninstall langchain langchain-core langchain-community langchain-ollama ollama` then `pip install --no-cache-dir -r requirements.txt`.
-   **Ollama Not Running/Models Not Pulled:**
    * Ensure the Ollama desktop application is running or the `ollama serve` command is active.
    * Use `ollama list` to see pulled models and `ollama pull <model_name>` to get them (e.g., `ollama pull llama3:8b`, `ollama pull nomic-embed-text`).
    * Check Ollama server logs for errors if models fail to load (especially with `format="json"`).
-   **Vector Store Not Found / "Core AI components failed to load":**
    * Make sure you have run `python src/preprocess.py` successfully after placing your PDFs in `docs_corpus/`.
    * Check that `vector_store/docs_faiss_index.faiss` and `vector_store/docs_faiss_index.pkl` exist.
    * Review console logs when starting Streamlit for more specific error messages from `src/utils.py` regarding model loading.
-   **Pydantic Validation Errors (e.g., "Field required", "Failed to parse ... from completion"):**
    * This means the LLM is not outputting JSON that matches the Pydantic model schema (`StructuredAnswer` or `FaithfulnessCheck`).
    * Check the "RAW LLM output" and "CLEANED LLM output" debug prints in your console (added in `generate_structured_answer` and `check_answer_faithfulness`). This will show you what the LLM actually produced.
    * The prompts in `rag_app.py` have been strengthened to guide the LLM, including providing examples of the JSON structure. If issues persist:
        * Try a different LLM model known for better JSON adherence and instruction following.
        * Further simplify the Pydantic models or the prompt instructions.
-   **No Relevant Documents Retrieved (High L2 Scores / "No documents met similarity threshold"):**
    * This is the most common issue for new, specialized documents.
    * **Use the "Similarity Score Threshold" slider in the UI.** Start high (e.g., 2.5-3.5) to see the raw scores in the debug expander. Then adjust.
    * **Experiment with `CHUNK_SIZE` and `CHUNK_OVERLAP` in `src/preprocess.py` and re-run it.** This is often the most impactful change.
    * Try very literal queries (copy-pasted sentences from your PDF).
    * If scores are still extremely high, consider if the chosen embedding model is suitable for your PDF's content.
-   **`TypeError: '>' not supported between instances of 'NoneType' and 'float'` or similar:**
    * This usually occurs when code tries to operate on a Pydantic field that might be `None` (e.g., `confidence_score`) without checking for `None` first. The latest version of `rag_app.py` should handle these for `confidence_score`. Review similar comparisons if new errors appear.

## Future Enhancements

-   **More Sophisticated Re-ranking:** Explore dedicated cross-encoder models.
-   **Parent Document Retriever (LangChain):** For better contextual understanding.
-   **Automated Evaluation Pipeline:** Integrate tools like RAGAs or DeepEval.
-   **UI for Parameter Tuning:** Expose more preprocessing or model parameters in the UI.
-   **Streaming LLM Responses:** For improved perceived responsiveness in the UI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
