"""
Utility functions for the RAG chatbot, including loading Ollama models
and cleaning up LLM-generated JSON strings.
These functions are designed to be Streamlit-agnostic for broader usability.
Caching is handled by the calling Streamlit application if needed.
"""
import json
import os
# Import from the new langchain-ollama package
from langchain_ollama import OllamaLLM # Use OllamaLLM for the language model
from langchain_ollama import OllamaEmbeddings # This name is correct

# --- Configuration ---
DEFAULT_OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
DEFAULT_OLLAMA_JSON_LLM_MODEL = os.getenv("OLLAMA_JSON_LLM_MODEL", DEFAULT_OLLAMA_LLM_MODEL)
DEFAULT_OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# --- Model Loading ---

def load_ollama_llm(model_name: str | None = None, format_json: bool = False) -> OllamaLLM | None:
    """
    Loads an Ollama Large Language Model using the langchain-ollama package.
    This function is Streamlit-agnostic.
    Caching should be handled by the caller if used in a Streamlit app.

    Args:
        model_name (str, optional): The name of the Ollama model to load.
        format_json (bool): If True, configures the LLM to output JSON.

    Returns:
        OllamaLLM | None: An instance of the Ollama LLM, or None if loading fails.
    """
    if model_name is None:
        model_to_load = DEFAULT_OLLAMA_JSON_LLM_MODEL if format_json else DEFAULT_OLLAMA_LLM_MODEL
    else:
        model_to_load = model_name

    try:
        llm_kwargs = {"model": model_to_load}
        if format_json:
            llm_kwargs["format"] = "json"

        # Using OllamaLLM from langchain_ollama
        llm = OllamaLLM(**llm_kwargs)

        if format_json:
            test_json_payload = '{"status":"ok"}'
            test_prompt_for_invoke = f"Output a simple JSON object: {test_json_payload}"
        else:
            test_prompt_for_invoke = "Hello!"

        _ = llm.invoke(test_prompt_for_invoke)
        print(f"Successfully loaded and tested Ollama LLM: {model_to_load} (JSON mode: {format_json}) from utils.py using langchain-ollama (OllamaLLM).")
        return llm
    except Exception as e:
        error_message = (
            f"Error initializing Ollama LLM '{model_to_load}' (JSON mode: {format_json}) in utils.py: {e}. "
            f"Ensure Ollama is running, the model '{model_to_load}' is pulled, "
            f"and the model supports JSON mode if requested. Check Ollama server logs."
        )
        print(f"ERROR: {error_message}")
        return None

def load_ollama_embeddings(model_name: str | None = None) -> OllamaEmbeddings | None:
    """
    Loads Ollama Embeddings using the langchain-ollama package.
    This function is Streamlit-agnostic.

    Args:
        model_name (str, optional): The name of the Ollama embedding model.

    Returns:
        OllamaEmbeddings | None: An instance of OllamaEmbeddings, or None if loading fails.
    """
    model_to_load = model_name if model_name else DEFAULT_OLLAMA_EMBEDDING_MODEL
    try:
        # Using OllamaEmbeddings from langchain_ollama
        embeddings = OllamaEmbeddings(model=model_to_load)
        _ = embeddings.embed_query("Test embedding model connectivity.")
        print(f"Successfully loaded and tested Ollama Embeddings: {model_to_load} from utils.py using langchain-ollama.")
        return embeddings
    except Exception as e:
        error_message = (
            f"Error initializing Ollama Embeddings '{model_to_load}' in utils.py: {e}. "
            f"Ensure Ollama is running and the model '{model_to_load}' is pulled."
        )
        print(f"ERROR: {error_message}")
        return None

# --- JSON Utilities ---
def cleanup_json_string(json_str: str) -> str:
    """
    Attempts to clean common issues in LLM-generated JSON strings to make them parsable.
    (Implementation remains the same)
    """
    if not isinstance(json_str, str):
        return ""
    s = json_str.strip()
    if s.startswith("```json"): s = s[7:]
    elif s.startswith("```"): s = s[3:]
    if s.endswith("```"): s = s[:-3]
    s = s.strip()
    first_brace = s.find('{'); last_brace = s.rfind('}')
    first_bracket = s.find('['); last_bracket = s.rfind(']')
    start_index = -1; end_index = -1
    if first_brace != -1 and last_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_index = first_brace; end_index = last_brace + 1
    elif first_bracket != -1 and last_bracket != -1:
        start_index = first_bracket; end_index = last_bracket + 1
    if start_index != -1 and end_index != -1 and start_index < end_index: s = s[start_index:end_index]
    s = s.replace(",\n]", "\n]").replace(",\n}", "\n}").replace(",]", "]").replace(",}", "}")
    return s.strip()

if __name__ == "__main__":
    print("--- Testing utils.py with langchain-ollama (OllamaLLM) ---")
    print("Testing Ollama LLM loading (text mode)...")
    test_llm_text_util = load_ollama_llm(format_json=False)
    if test_llm_text_util:
        print(f"  Text LLM ({test_llm_text_util.model}) loaded. Test invoke: {test_llm_text_util.invoke('Say hi.')}")
    else:
        print("  Failed to load text LLM.")

    print("\nTesting Ollama LLM loading (JSON mode)...")
    test_llm_json_util = load_ollama_llm(format_json=True)
    if test_llm_json_util:
        json_payload_for_test = '{"key":"value"}'
        invoke_prompt = f'Output JSON: {json_payload_for_test}'
        print(f"  JSON LLM ({test_llm_json_util.model}) loaded. Test invoke: {test_llm_json_util.invoke(invoke_prompt)}")
    else:
        print("  Failed to load JSON LLM.")

    print("\nTesting Ollama Embeddings loading...")
    test_embeddings_util = load_ollama_embeddings()
    if test_embeddings_util:
        print(f"  Embeddings ({test_embeddings_util.model}) loaded. Test embed length: {len(test_embeddings_util.embed_query('test'))}")
    else:
        print("  Failed to load embeddings.")
    print("--- End of utils.py tests ---")
