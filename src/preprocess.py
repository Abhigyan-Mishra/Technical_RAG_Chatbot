"""
Preprocessing script for the RAG chatbot.
This script performs the following actions:
1.  Locates PDF documents in the specified corpus directory.
2.  Loads and extracts text content from each PDF.
3.  Adds source filename metadata to each document chunk.
4.  Splits the extracted text into smaller, manageable chunks.
5.  Generates vector embeddings for each chunk using a local Ollama embedding model.
6.  Creates a FAISS vector store from these embeddings.
7.  Saves the FAISS index and associated corpus metadata to disk for later use by the chatbot application.

This script should be run once (or whenever the source documents change) before starting
the chatbot application.
"""
import os
import glob
import pickle
import sys

# Add the 'src' directory to Python's path to allow sibling imports
# This is useful if running the script directly from the 'src' directory or project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import load_ollama_embeddings # Import from local utils module

# --- Configuration ---
# Determine the base directory of the project (assuming this script is in src/)
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCS_CORPUS_DIR = os.path.join(PROJECT_ROOT_DIR, "docs_corpus")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT_DIR, "vector_store")

# Embedding model name (can be overridden if needed, otherwise uses default from utils.py)
# OLLAMA_EMBEDDING_MODEL_OVERRIDE = "your_specific_embedding_model_here"

# Chunking parameters - CRITICAL for RAG performance. Tune these based on your document characteristics.
CHUNK_SIZE = 1000  # Target size of each text chunk in characters/tokens (depends on splitter)
CHUNK_OVERLAP = 200 # Number of characters/tokens to overlap between consecutive chunks

# Filenames for the saved vector store and metadata
VECTORSTORE_INDEX_FILENAME = "docs_faiss_index" # The actual index files will have extensions like .faiss, .pkl
CORPUS_METADATA_FILENAME = "docs_corpus_metadata.pkl"

def create_and_save_vectorstore(
    docs_directory: str = DOCS_CORPUS_DIR,
    vector_store_output_directory: str = VECTOR_STORE_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model_name: str | None = None # If None, uses default from utils.py
):
    """
    Processes PDF documents, creates a FAISS vector store, and saves it.

    Args:
        docs_directory (str): Path to the directory containing PDF documents.
        vector_store_output_directory (str): Path to the directory where the vector store will be saved.
        chunk_size (int): The target size for text chunks.
        chunk_overlap (int): The overlap between text chunks.
        embedding_model_name (str, optional): Specific Ollama embedding model to use.
                                              Defaults to the one specified in utils.py.
    """
    print(f"--- Starting Document Preprocessing ---")
    print(f"Source PDF documents directory: '{docs_directory}'")
    print(f"Vector store output directory: '{vector_store_output_directory}'")
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

    pdf_files = glob.glob(os.path.join(docs_directory, "*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{docs_directory}'. Please add your PDF documents.")
        return

    print(f"\nFound {len(pdf_files)} PDF document(s) to process:")
    for pdf_file in pdf_files:
        print(f"  - {os.path.basename(pdf_file)}")

    all_document_pages = [] # Stores LangChain Document objects from all PDFs
    corpus_metadata_summary = [] # Stores summary metadata about each processed PDF

    for doc_path in pdf_files:
        doc_basename = os.path.basename(doc_path)
        print(f"\nProcessing: '{doc_basename}'...")
        try:
            loader = PyMuPDFLoader(doc_path)
            pages = loader.load() # Each page is a LangChain Document

            # Add source filename to metadata for each page
            for page_doc in pages:
                page_doc.metadata["source_filename"] = doc_basename
                # PyMuPDFLoader usually includes 'page' (0-indexed) in metadata
                if 'page' in page_doc.metadata:
                    page_doc.metadata["page_number_0_indexed"] = page_doc.metadata['page']
                else: # Fallback if page number isn't directly available
                    page_doc.metadata["page_number_0_indexed"] = -1 # Indicate unknown

            all_document_pages.extend(pages)
            corpus_metadata_summary.append({
                "filename": doc_basename,
                "initial_page_count": len(pages)
            })
            print(f"  Successfully loaded {len(pages)} pages from '{doc_basename}'.")
        except Exception as e:
            print(f"  Error loading or processing '{doc_basename}': {e}")
            continue # Skip to the next document

    if not all_document_pages:
        print("\nError: No documents were successfully loaded or processed. Aborting preprocessing.")
        return

    print(f"\nTotal pages/initial documents loaded from all PDFs: {len(all_document_pages)}")

    print("\nSplitting documents into smaller text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use character length for splitting
        add_start_index=True # Adds start index of chunk in original document to metadata
    )
    final_text_chunks = text_splitter.split_documents(all_document_pages)

    if not final_text_chunks:
        print("Error: Failed to split documents into text chunks. Aborting.")
        return

    print(f"  Successfully split into {len(final_text_chunks)} text chunks.")
    # Example: print metadata of a few chunks to verify
    # for i, chunk in enumerate(final_text_chunks[:2]):
    # print(f"    Chunk {i} metadata: {chunk.metadata}")

    print("\nLoading Ollama embedding model...")
    # Use the provided model name or let load_ollama_embeddings use its default
    embeddings = load_ollama_embeddings(model_name=embedding_model_name)
    if not embeddings:
        print("Error: Failed to load Ollama embedding model. Aborting preprocessing.")
        return
    print(f"  Using embedding model: {embeddings.model}")

    print("\nGenerating embeddings and creating FAISS vector store...")
    try:
        # Ensure all chunks have non-empty page_content before sending to FAISS
        valid_chunks = [chunk for chunk in final_text_chunks if chunk.page_content and chunk.page_content.strip()]
        if not valid_chunks:
            print("Error: All document chunks are empty after filtering. Cannot create vector store.")
            return
        if len(valid_chunks) < len(final_text_chunks):
            print(f"  Warning: {len(final_text_chunks) - len(valid_chunks)} chunks were empty and have been excluded.")

        vectorstore = FAISS.from_documents(valid_chunks, embeddings)
        print(f"  FAISS vector store created successfully with {vectorstore.index.ntotal} vectors (chunks).")
    except Exception as e:
        print(f"  Error creating FAISS vector store: {e}")
        return

    # Ensure the output directory exists
    os.makedirs(vector_store_output_directory, exist_ok=True)

    vectorstore_save_path = os.path.join(vector_store_output_directory, VECTORSTORE_INDEX_FILENAME)
    metadata_save_path = os.path.join(vector_store_output_directory, CORPUS_METADATA_FILENAME)

    print(f"\nSaving FAISS vector store to: '{vectorstore_save_path}'")
    try:
        vectorstore.save_local(vectorstore_save_path)
        print("  Vector store saved.")
    except Exception as e:
        print(f"  Error saving vector store: {e}")
        return

    print(f"Saving corpus metadata to: '{metadata_save_path}'")
    try:
        # Add overall processing stats to metadata
        processing_summary = {
            "total_pdfs_processed": len(corpus_metadata_summary),
            "total_chunks_generated": len(final_text_chunks),
            "chunk_size_setting": chunk_size,
            "chunk_overlap_setting": chunk_overlap,
            "embedding_model_used": embeddings.model,
            "detailed_file_summaries": corpus_metadata_summary
        }
        with open(metadata_save_path, 'wb') as f:
            pickle.dump(processing_summary, f)
        print("  Corpus metadata saved.")
    except Exception as e:
        print(f"  Error saving corpus metadata: {e}")
        return

    print("\n--- Document Preprocessing Complete ---")

if __name__ == "__main__":
    print("========================================================")
    print(" Advanced RAG Chatbot - Document Preprocessing Script ")
    print("========================================================")
    print("This script will process PDFs from 'docs_corpus/' and create a vector store in 'vector_store/'.")
    print("Ensure your Ollama server is running and the required embedding model is pulled.")
    print(f"Default embedding model (from utils.py): {load_ollama_embeddings().model if load_ollama_embeddings() else 'Not configured'}")
    print(f"Default chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print("--------------------------------------------------------")

    # Example of how to make it configurable via command line arguments (optional)
    # import argparse
    # parser = argparse.ArgumentParser(description="Preprocess PDF documents for RAG chatbot.")
    # parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Target chunk size.")
    # parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap.")
    # parser.add_argument("--embedding_model", type=str, default=None, help="Ollama embedding model name.")
    # args = parser.parse_args()
    # create_and_save_vectorstore(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, embedding_model_name=args.embedding_model)

    create_and_save_vectorstore()
