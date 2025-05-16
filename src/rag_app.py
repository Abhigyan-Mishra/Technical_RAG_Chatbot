"""
Main Streamlit application for the Advanced RAG Chatbot.
(Caching applied within this script for Streamlit context)
Debug information now persists in session_state.
TypeError for NoneType comparison with float fixed.
Corrected f-string for confidence score display.
Strengthened JSON prompt for answer generation & added more context debugging.
Revised FaithfulnessCheck prompt.
"""
import streamlit as st
import os
import pickle
import json
import sys
from typing import List, Dict, Any

# Add the 'src' directory to Python's path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
# Explicitly import types from the new langchain-ollama package
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings


# Import local modules
from pydantic_models import StructuredAnswer, FaithfulnessCheck, AnswerReference
# Import the raw utility functions
from utils import load_ollama_llm as util_load_ollama_llm, \
                  load_ollama_embeddings as util_load_ollama_embeddings, \
                  cleanup_json_string, \
                  DEFAULT_OLLAMA_LLM_MODEL, DEFAULT_OLLAMA_JSON_LLM_MODEL, DEFAULT_OLLAMA_EMBEDDING_MODEL

# --- Configuration ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT_DIR, "vector_store")
VECTORSTORE_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "docs_faiss_index")
CORPUS_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "docs_corpus_metadata.pkl")

INITIAL_RETRIEVAL_K_DEFAULT = 10
RE_RANKED_K_DEFAULT = 3
SIMILARITY_SCORE_THRESHOLD_DEFAULT = 2.8

# --- Cached Model Loading for Streamlit App ---
@st.cache_resource
def get_cached_ollama_llm(model_name: str | None = None, format_json: bool = False) -> OllamaLLM | None:
    return util_load_ollama_llm(model_name, format_json)

@st.cache_resource
def get_cached_ollama_embeddings(model_name: str | None = None) -> OllamaEmbeddings | None:
    return util_load_ollama_embeddings(model_name)

@st.cache_resource
def get_cached_vector_store(_embeddings_model_to_pass_to_faiss: OllamaEmbeddings | None, _vector_store_path: str) -> FAISS | None:
    if not _embeddings_model_to_pass_to_faiss:
        print("ERROR in get_cached_vector_store: Embedding model is None.")
        return None
    if not os.path.exists(_vector_store_path):
        print(f"ERROR in get_cached_vector_store: Vector store not found at '{_vector_store_path}'.")
        return None
    try:
        return FAISS.load_local(
            _vector_store_path,
            _embeddings_model_to_pass_to_faiss,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"ERROR in get_cached_vector_store: Error loading vector store: {e}")
        return None

# --- Core RAG Functions ---

def expand_query(query: str, llm: OllamaLLM) -> List[str]:
    # ... (Implementation from src_rag_app_py_v10_prompt_fix remains the same) ...
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""Given the user's technical query: "{query}"
Generate 2-3 diverse rephrased versions of this query that could help find relevant information in technical manuals.
Focus on synonyms, breaking down complex parts, or different search angles.
Your response MUST be ONLY a valid JSON list of strings. For example: ["device factory reset procedure", "steps to reset Model X", "troubleshoot device reset"]
Do NOT output a JSON dictionary or any other text.
Output:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response_str = "[]"
    try:
        response_str = chain.invoke({"query": query})['text']
        cleaned_response_str = cleanup_json_string(response_str)
        parsed_json = json.loads(cleaned_response_str)
        expanded_queries = []
        if isinstance(parsed_json, list) and all(isinstance(q, str) for q in parsed_json):
            expanded_queries = parsed_json
        elif isinstance(parsed_json, dict):
            print(f"WARNING: Query expansion returned a dictionary: {parsed_json}. Attempting to extract query list.")
            for val in parsed_json.values():
                if isinstance(val, list) and all(isinstance(q, str) for q in val):
                    expanded_queries.extend(val); break
            if not expanded_queries: print(f"WARNING: Could not extract list from query expansion dict.")
        if expanded_queries:
            return list(dict.fromkeys([query.strip()] + [q.strip() for q in expanded_queries if q.strip()]))
        else:
            print(f"WARNING: Query expansion did not yield valid list from: {cleaned_response_str}. Using original query.")
            return [query]
    except Exception as e:
        response_str_snippet = response_str[:300] if 'response_str' in locals() else "N/A"
        print(f"WARNING: Error in query expansion (LLM: {llm.model}): {e}. Using original. LLM output: '{response_str_snippet}'")
        return [query]


def retrieve_and_rerank_chunks(
    queries: List[str],
    vectorstore: FAISS,
    llm_reranker: OllamaLLM,
    top_n_initial: int,
    top_n_final: int,
    similarity_threshold: float
) -> tuple[List[Document], List[str]]:
    # ... (Implementation from src_rag_app_py_v10_prompt_fix remains the same) ...
    debug_messages: List[str] = []
    all_retrieved_docs_with_scores_before_threshold: List[tuple[Document, float]] = []
    debug_messages.append("**Initial Retrieval (Before Threshold & Deduplication):**")
    for i, q_variant in enumerate(queries):
        retrieved_with_scores = vectorstore.similarity_search_with_score(q_variant, k=top_n_initial)
        debug_messages.append(f"**Query Variant {i+1}:** `{q_variant}` (Retrieved {len(retrieved_with_scores)} chunks initially)")
        for doc_idx, (doc, score) in enumerate(retrieved_with_scores):
            all_retrieved_docs_with_scores_before_threshold.append((doc, score))
            debug_messages.append(f"  - Raw Doc {doc_idx+1} (Score: `{score:.4f}`): {doc.page_content[:100]}...")
    unique_docs_best_scores: Dict[str, tuple[Document, float]] = {}
    for doc, score in all_retrieved_docs_with_scores_before_threshold:
        if doc.page_content not in unique_docs_best_scores or score < unique_docs_best_scores[doc.page_content][1]:
            unique_docs_best_scores[doc.page_content] = (doc, score)
    filtered_docs_after_threshold: List[tuple[Document, float]] = []
    for doc, score in unique_docs_best_scores.values():
        if score < similarity_threshold:
            doc.metadata['retrieval_score'] = score
            filtered_docs_after_threshold.append((doc, score))
    sorted_unique_docs_tuples = sorted(filtered_docs_after_threshold, key=lambda x: x[1])
    unique_docs_for_reranking = [doc_tuple[0] for doc_tuple in sorted_unique_docs_tuples]
    debug_messages.append("---")
    debug_messages.append(f"**Chunks After Similarity Threshold (`{similarity_threshold}`) & Deduplication (to be re-ranked):** {len(unique_docs_for_reranking)} chunks")
    for i, doc_item in enumerate(unique_docs_for_reranking):
        debug_messages.append(f"  - Potential Rank {i+1} (Score: `{doc_item.metadata.get('retrieval_score', 'N/A'):.4f}`): {doc_item.page_content[:100]}...")
    if not unique_docs_for_reranking:
        print("INFO: No documents met the similarity threshold or survived deduplication.")
        return [], debug_messages
    if len(unique_docs_for_reranking) <= top_n_final:
        print(f"INFO: Returning {len(unique_docs_for_reranking)} documents as not enough for full re-ranking.")
        debug_messages.append(f"INFO: Re-ranking skipped, too few docs ({len(unique_docs_for_reranking)} <= {top_n_final}).")
        return unique_docs_for_reranking, debug_messages
    context_for_reranking = ""
    doc_map_for_reranker = {i: doc_item for i, doc_item in enumerate(unique_docs_for_reranking)}
    for i, doc_item in doc_map_for_reranker.items():
        source_file = doc_item.metadata.get('source_filename', 'Unknown Source')
        page_num_0_indexed = doc_item.metadata.get('page_number_0_indexed', -1)
        page_display = page_num_0_indexed + 1 if page_num_0_indexed != -1 else "N/A"
        context_for_reranking += f"Chunk_ID: {i}\nSource: {source_file}, Page: {page_display}\nContent Snippet: {doc_item.page_content[:300]}...\n\n---\n\n"
    rerank_prompt_template = PromptTemplate(
        input_variables=["original_query", "context_chunks", "top_n_final"],
        template="""You are a relevance ranking AI... (rest of prompt from v7) ... Output ONLY a JSON list of integer Chunk_IDs. Example: [3, 0, 5]"""
    )
    rerank_chain = LLMChain(llm=llm_reranker, prompt=rerank_prompt_template)
    response_str = "[]"
    try:
        response_str = rerank_chain.invoke({
            "original_query": queries[0], "context_chunks": context_for_reranking, "top_n_final": top_n_final
        })['text']
        cleaned_response_str = cleanup_json_string(response_str)
        reranked_ids = json.loads(cleaned_response_str)
        if isinstance(reranked_ids, list) and all(isinstance(id_val, int) for id_val in reranked_ids):
            reranked_docs = [doc_map_for_reranker[id_val] for id_val in reranked_ids if id_val in doc_map_for_reranker]
            debug_messages.append("---")
            debug_messages.append(f"**Re-ranked Chunk IDs (from LLM):** `{reranked_ids}`")
            debug_messages.append(f"**Final {len(reranked_docs)} Chunks After Re-ranking:**")
            for i, doc_item_final in enumerate(reranked_docs[:top_n_final]):
                debug_messages.append(f"  - Final Rank {i+1} (Orig Score: `{doc_item_final.metadata.get('retrieval_score', 'N/A'):.4f}`): {doc_item_final.page_content[:100]}...")
            return reranked_docs[:top_n_final], debug_messages
        else:
            print(f"WARNING: Re-ranking LLM returned an unexpected format for IDs: {reranked_ids}. Falling back.")
            debug_messages.append(f"WARNING: Re-ranking LLM returned unexpected format for IDs: {reranked_ids}. Falling back to top {top_n_final} pre-reranked.")
            return unique_docs_for_reranking[:top_n_final], debug_messages
    except Exception as e:
        response_str_snippet = response_str[:300] if 'response_str' in locals() else "N/A"
        print(f"WARNING: Error in LLM-based re-ranking (LLM: {llm_reranker.model}): {e}. Falling back. LLM output: '{response_str_snippet}'")
        debug_messages.append(f"WARNING: Error in LLM-based re-ranking: {e}. Falling back to top {top_n_final} pre-reranked.")
        return unique_docs_for_reranking[:top_n_final], debug_messages

def generate_structured_answer(query: str, context_docs: List[Document], llm: OllamaLLM, debug_container) -> StructuredAnswer | None:
    # ... (Implementation from src_rag_app_py_v10_prompt_fix remains the same) ...
    if not context_docs:
        print("INFO: No context documents provided to generate_structured_answer.")
        with debug_container:
            st.warning("No context documents were passed to `generate_structured_answer` function.")
        return StructuredAnswer(
            comprehensive_answer="I could not find any relevant information in the provided documentation to answer your query.",
            detailed_references=[], confidence_score=0.0,
            issues_or_missing_info="No relevant document sections were found or passed for answer generation."
        )
    context_str = ""
    with debug_container:
        st.markdown("---")
        st.markdown(f"**Context Passed to `generate_structured_answer` (LLM: {llm.model}):**")
        st.markdown(f"**User Query:** `{query}`")
        st.markdown(f"**Number of Context Chunks:** {len(context_docs)}")
    for i, doc_item in enumerate(context_docs):
        source_file = doc_item.metadata.get('source_filename', 'Unknown Source')
        page_num_0_indexed = doc_item.metadata.get('page_number_0_indexed', -1)
        page_display = page_num_0_indexed + 1 if page_num_0_indexed != -1 else "N/A"
        context_block_header = f"Context_Block_ID: {i}\nSource_Filename: {source_file}\nPage_Number: {page_display}\nContent:\n"
        context_str += context_block_header + doc_item.page_content + "\n\n---\n\n"
        with debug_container:
            st.markdown(f"**{context_block_header.strip()}**")
            st.caption(doc_item.page_content[:300] + "...")
    parser = PydanticOutputParser(pydantic_object=StructuredAnswer)
    output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    answer_prompt_template_str = """You are an AI assistant specialized in answering questions from technical documentation.
Your response MUST be based *EXCLUSIVELY* on the provided "Context from Documentation" below.
Do NOT use any external knowledge or make assumptions beyond the provided text.
Your entire output MUST be a single, valid JSON object matching the provided schema. Do not add any explanatory text before or after the JSON object.

**User Query:**
"{query}"

**Context from Documentation (each block has a Context_Block_ID, Source_Filename, and Page_Number):**
--- START OF PROVIDED CONTEXT ---
{context}
--- END OF PROVIDED CONTEXT ---

**Your Task:**
Generate a JSON response that strictly adheres to the following Pydantic model format and instructions.
The JSON schema you MUST follow is:
{format_instructions}

**Example of Expected JSON Output Structure (This is an EXAMPLE of the *structure*, fill with actual data based on the PROVIDED CONTEXT above):**
```json
{{
  "comprehensive_answer": "The main purpose of the quantum component is to tackle complex optimization problems and elevate analytical insights within portfolio management and trading strategy optimization.",
  "detailed_references": [
    {{
      "statement": "The quantum component tackles complex optimization problems.",
      "source_filename": "Rune_Technical_Blueprint_Quantum.pdf",
      "page_number": 2,
      "verbatim_quote": "Its primary function is to tackle complex optimization problems..."
    }},
    {{
      "statement": "It aims to elevate analytical insights in portfolio management.",
      "source_filename": "Rune_Technical_Blueprint_Quantum.pdf",
      "page_number": 2,
      "verbatim_quote": "...and elevate analytical insights within the domains of portfolio management..."
    }}
  ],
  "confidence_score": 0.9,
  "issues_or_missing_info": "None"
}}
```
If the PROVIDED CONTEXT does not contain information to answer the query, `comprehensive_answer` should clearly state this (e.g., "The provided documentation does not contain specific information about 'Post Quantum'."), `detailed_references` should be an empty list, `confidence_score` should be low (e.g., 0.0 or 0.1), and `issues_or_missing_info` should explain why an answer cannot be provided from the context.

Output ONLY the JSON object as described.
"""
    prompt = PromptTemplate(
        template=answer_prompt_template_str,
        input_variables=["query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response_str = "{}"
    try:
        print(f"Invoking LLM for structured answer with query: '{query[:100]}...' and {len(context_docs)} context_docs.")
        response_str = chain.invoke({"query": query, "context": context_str})['text']
        cleaned_response_str = cleanup_json_string(response_str)
        print(f"\n--- RAW LLM output for structured answer ---\n{response_str}\n---")
        print(f"--- CLEANED LLM output for structured answer ---\n{cleaned_response_str}\n---")
        parsed_output = parser.parse(cleaned_response_str)
        print("INFO: Successfully parsed structured answer from LLM.")
        return parsed_output
    except Exception as e_parse:
        print(f"WARNING: Initial Pydantic parsing failed for structured answer: {e_parse}. Attempting to fix with OutputFixingParser.")
        try:
            fixed_output = output_fixing_parser.parse(cleaned_response_str)
            print("INFO: Successfully fixed and parsed structured answer using OutputFixingParser.")
            return fixed_output
        except Exception as e_fix:
            error_msg_details = f"Error generating/parsing structured answer even after attempting to fix (LLM: {llm.model}): {e_fix}. Original parsing error: {e_parse}. Cleaned LLM output snippet: '{cleaned_response_str[:500]}...'"
            print(f"ERROR: {error_msg_details}")
            return StructuredAnswer(
                comprehensive_answer="An error occurred while processing the AI's response. The output format was not as expected.",
                detailed_references=[], confidence_score=0.0,
                issues_or_missing_info=f"Internal parsing/fixing error. Details: {str(e_fix)[:200]}"
            )

def check_answer_faithfulness(
    structured_answer_obj: StructuredAnswer,
    context_docs: List[Document],
    llm: OllamaLLM
) -> FaithfulnessCheck | None:
    """
    Uses an LLM to check if the generated structured answer is faithful to the provided context.
    """
    if not structured_answer_obj or not structured_answer_obj.comprehensive_answer or \
       "error" in structured_answer_obj.comprehensive_answer.lower() or \
       not structured_answer_obj.detailed_references: # If no references, hard to check faithfulness properly
        return FaithfulnessCheck(
            is_faithful=False,
            explanation="Answer generation failed, indicated an error, or provided no references to check against.",
            conflicting_info_present=False,
            unsupported_statements=["Original answer was problematic or had no references."]
        )

    context_str = "\n\n---\n\n".join(
        [f"Source: {d.metadata.get('source_filename', 'N/A')}, Page: {d.metadata.get('page_number_0_indexed', -1)+1 if d.metadata.get('page_number_0_indexed', -1) != -1 else 'N/A'}\n{d.page_content}" for d in context_docs]
    )
    answer_to_check = f"Main Answer: {structured_answer_obj.comprehensive_answer}\n\nReferenced Statements:\n"
    for i, ref in enumerate(structured_answer_obj.detailed_references):
        answer_to_check += f"{i+1}. Statement: \"{ref.statement}\" (Claimed source: {ref.source_filename}, Page: {ref.page_number}, Quote: \"{ref.verbatim_quote}\")\n"


    parser = PydanticOutputParser(pydantic_object=FaithfulnessCheck)
    output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm) # LLM for fixing should be JSON capable

    # Strengthened prompt for FaithfulnessCheck
    faithfulness_prompt_template_str = """You are an AI evaluator. Your task is to assess if the "Generated Answer to Evaluate" is FULLY and ACCURATELY supported by the "Retrieved Context".
The answer should NOT introduce any information not present in the context. Minor rephrasing is acceptable if meaning is preserved.
Your entire output MUST be a single, valid JSON object matching the provided schema. Do not add any explanatory text before or after the JSON object.

**Retrieved Context:**
--- START OF CONTEXT ---
{retrieved_context}
--- END OF CONTEXT ---

**Generated Answer to Evaluate (includes main answer and its referenced statements):**
--- START OF ANSWER ---
{answer_text_to_evaluate}
--- END OF ANSWER ---

**Your Task:**
Generate a JSON response that strictly adheres to the following Pydantic model format and instructions.
The JSON schema you MUST follow is:
{format_instructions}

**Example of Expected JSON Output Structure (This is an EXAMPLE of the *structure*, fill with actual data based on the evaluation):**
```json
{{
  "is_faithful": true,
  "explanation": "The answer is well-supported by the provided context, and all referenced statements align with the quotes and source information.",
  "conflicting_info_present": false,
  "unsupported_statements": []
}}
```
Or, if not faithful:
```json
{{
  "is_faithful": false,
  "explanation": "The statement 'XYZ' in the answer is not directly supported by the provided context.",
  "conflicting_info_present": false,
  "unsupported_statements": ["XYZ"]
}}
```

**Instructions for filling the JSON (refer to the schema above):**
1.  `is_faithful`: `true` if ALL parts of the "Generated Answer to Evaluate" (both the main answer and all referenced statements) are directly and verifiably supported by the "Retrieved Context". `false` otherwise.
2.  `explanation`: Provide a brief explanation for your assessment. If not faithful, specify WHICH PART of the answer is unsupported or contradicts the context.
3.  `conflicting_info_present`: `true` if you noticed any information within the "Retrieved Context" itself that seems to conflict regarding the user's likely query.
4.  `unsupported_statements`: If `is_faithful` is `false`, list the specific statements from the "Generated Answer to Evaluate" that are not supported by the context. If faithful, this should be an empty list (`[]`).

Output ONLY the JSON object as described.
"""
    prompt = PromptTemplate(
        template=faithfulness_prompt_template_str,
        input_variables=["answer_text_to_evaluate", "retrieved_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response_str = "{}"
    try:
        print(f"Invoking LLM for faithfulness check with answer: '{structured_answer_obj.comprehensive_answer[:100]}...'")
        response_str = chain.invoke({"answer_text_to_evaluate": answer_to_check, "retrieved_context": context_str})['text']
        cleaned_response_str = cleanup_json_string(response_str)
        
        print(f"\n--- RAW LLM output for faithfulness check ---\n{response_str}\n---")
        print(f"--- CLEANED LLM output for faithfulness check ---\n{cleaned_response_str}\n---")

        return parser.parse(cleaned_response_str)
    except Exception as e_parse:
        print(f"WARNING: Initial Pydantic parsing failed for faithfulness check: {e_parse}. Attempting to fix.")
        try:
            return output_fixing_parser.parse(cleaned_response_str)
        except Exception as e_fix:
            error_msg_details = f"Error parsing faithfulness check (LLM: {llm.model}): {e_fix}. Cleaned LLM output: '{cleaned_response_str[:500]}...'"
            print(f"ERROR: {error_msg_details}")
            return FaithfulnessCheck(
                is_faithful=False,
                explanation=f"Error parsing faithfulness check response. Details: {str(e_fix)[:200]}",
                conflicting_info_present=False,
                unsupported_statements=["Faithfulness check LLM response parsing failed."]
            )

# --- Streamlit UI Setup ---
# (The UI part remains largely the same as in src_rag_app_py_v9_typeerror_fix.
#  Key is that the main chat loop correctly calls the updated generate_structured_answer and check_answer_faithfulness)
# For brevity, the full UI code is not repeated here but should be copied from the previous version.
# ... (Copy the Streamlit UI setup and main chat interface logic from src_rag_app_py_v9_typeerror_fix here)
st.set_page_config(layout="wide", page_title="Advanced Technical RAG Chatbot")
st.title("🛠️ Advanced RAG Chatbot for Technical Documentation")

st.sidebar.header("⚙️ Retrieval Configuration")
similarity_threshold_ui = st.sidebar.slider(
    "Similarity Score Threshold (L2 Distance - lower is more similar)",
    min_value=0.1, max_value=4.0, value=SIMILARITY_SCORE_THRESHOLD_DEFAULT, step=0.05,
    help="Lower values make retrieval stricter. Adjust based on observed scores in debug output."
)
initial_k_ui = st.sidebar.slider(
    "Initial Chunks to Retrieve (per query variant)",
    min_value=1, max_value=25, value=INITIAL_RETRIEVAL_K_DEFAULT, step=1
)
rerank_k_ui = st.sidebar.slider(
    "Final Chunks to Use (after re-ranking)",
    min_value=1, max_value=10, value=RE_RANKED_K_DEFAULT, step=1
)

st.sidebar.header("📚 Indexed Corpus Overview")
if os.path.exists(CORPUS_METADATA_PATH):
    try:
        with open(CORPUS_METADATA_PATH, 'rb') as f: corpus_metadata = pickle.load(f)
        st.sidebar.json(corpus_metadata, expanded=False)
    except Exception as e: st.sidebar.warning(f"Could not load corpus metadata: {e}")
else: st.sidebar.warning(f"Corpus metadata file not found: '{CORPUS_METADATA_PATH}'. Run 'preprocess.py'.")

llm_for_json_outputs = get_cached_ollama_llm(model_name=DEFAULT_OLLAMA_JSON_LLM_MODEL, format_json=True)
llm_for_query_tasks = get_cached_ollama_llm(model_name=DEFAULT_OLLAMA_JSON_LLM_MODEL, format_json=True)
embeddings_model_instance = get_cached_ollama_embeddings()
vector_store_instance = None
if embeddings_model_instance:
    vector_store_instance = get_cached_vector_store(embeddings_model_instance, VECTORSTORE_INDEX_PATH)

st.caption(f"Using LLMs: `{DEFAULT_OLLAMA_JSON_LLM_MODEL}` (JSON), `{DEFAULT_OLLAMA_LLM_MODEL}` (General) | Embeddings: `{DEFAULT_OLLAMA_EMBEDDING_MODEL}`")

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_debug_messages" not in st.session_state: st.session_state.last_debug_messages = []

debug_expander = st.expander("Retrieval & Re-ranking Debug Info (Last Query)", expanded=False)
with debug_expander:
    if st.session_state.last_debug_messages:
        for msg in st.session_state.last_debug_messages: st.markdown(msg)
    else: st.caption("No query processed yet in this session, or no debug messages were generated.")

for message_data in st.session_state.chat_history:
    with st.chat_message(message_data["role"]):
        st.markdown(message_data["content"])
        if message_data["role"] == "assistant" and "structured_answer_obj" in message_data:
            sa: StructuredAnswer = message_data["structured_answer_obj"]
            if "faithfulness_check_obj" in message_data and message_data["faithfulness_check_obj"]:
                fc: FaithfulnessCheck = message_data["faithfulness_check_obj"]
                if fc.is_faithful: st.success(f"✅ Faithfulness: Consistent. {fc.explanation}")
                else: st.warning(f"⚠️ Faithfulness: Potential inconsistency! {fc.explanation}")
                if fc.conflicting_info_present: st.info("ℹ️ Context might have conflicting info.")
                if fc.unsupported_statements:
                    with st.expander("Unsupported Statements (Faithfulness Check)"):
                        for us_stmt in fc.unsupported_statements: st.markdown(f"- {us_stmt}")
            
            confidence_display_text = "N/A"
            if sa.confidence_score is not None:
                try:
                    confidence_display_text = f"{sa.confidence_score*100:.1f}"
                except TypeError: 
                    confidence_display_text = "Error calculating"
            st.info(f"📊 LLM Confidence: {confidence_display_text}%")

            if sa.issues_or_missing_info and sa.issues_or_missing_info.lower() != "none":
                st.warning(f"🔍 LLM Issues/Missing Info: {sa.issues_or_missing_info}")
            with st.expander("🔍 Detailed References & Quotes", expanded=False):
                if not sa.detailed_references: st.write("No specific references provided.")
                else:
                    for i, ref in enumerate(sa.detailed_references):
                        st.markdown(f"**Ref {i+1}:** {ref.statement}")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;↳ **Source:** `{ref.source_filename}`, Page: `{ref.page_number}`")
                        st.caption(f"&nbsp;&nbsp;&nbsp;↳ **Quote:** \"{ref.verbatim_quote}\"")
                        if i < len(sa.detailed_references) - 1: st.markdown("---")
        st.markdown("---")

if not llm_for_json_outputs or not llm_for_query_tasks or not embeddings_model_instance or not vector_store_instance:
    st.error("Core AI components failed to load. Chatbot is disabled. Check console logs, Ollama setup, and ensure 'preprocess.py' has run successfully.")
else:
    user_query = st.chat_input("Ask a question about your technical documentation:")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)
        
        with st.spinner("Processing your query... (This may take a moment)"):
            current_query_debug_messages = []
            final_answer_obj = None; faithfulness_obj = None
            
            st.session_state.last_debug_messages = [] 

            expanded_queries = expand_query(user_query, llm_for_query_tasks)
            current_query_debug_messages.append("**Expanded Queries:**")
            current_query_debug_messages.append(f"```json\n{json.dumps(expanded_queries, indent=2)}\n```")
            
            relevant_chunks, retrieval_debug_msgs = retrieve_and_rerank_chunks(
                expanded_queries, vector_store_instance, llm_for_query_tasks,
                initial_k_ui, rerank_k_ui, similarity_threshold_ui
            )
            current_query_debug_messages.extend(retrieval_debug_msgs)
            
            if not relevant_chunks:
                print("WARNING: No relevant information found after retrieval and re-ranking.")
                final_answer_obj = StructuredAnswer(
                    comprehensive_answer="I could not find any relevant information in the provided documentation to answer your query after extensive searching.",
                    detailed_references=[], confidence_score=0.0,
                    issues_or_missing_info="No relevant document sections were retrieved that met the similarity threshold or survived re-ranking."
                )
                current_query_debug_messages.append("WARNING: No relevant chunks found to generate an answer.")
            else:
                current_query_debug_messages.append("---")
                current_query_debug_messages.append(f"**Final {len(relevant_chunks)} Chunks Passed to Answer Generation:**")
                for i, chunk_doc in enumerate(relevant_chunks):
                    page_num_0_idx = chunk_doc.metadata.get('page_number_0_indexed', -1)
                    page_display = page_num_0_idx + 1 if page_num_0_idx != -1 else "N/A"
                    ret_score_val = chunk_doc.metadata.get('retrieval_score', float('inf'))
                    ret_score_str = f"{ret_score_val:.4f}" if isinstance(ret_score_val, float) else str(ret_score_val)
                    current_query_debug_messages.append(f"  - Chunk {i+1} (Source: {chunk_doc.metadata.get('source_filename','N/A')}, Page: {page_display}) (Score: {ret_score_str}): {chunk_doc.page_content[:100]}...")
                
                final_answer_obj = generate_structured_answer(user_query, relevant_chunks, llm_for_json_outputs, debug_container=debug_expander) # Pass debug_expander
                
                if final_answer_obj and final_answer_obj.confidence_score is not None and final_answer_obj.confidence_score > 0.01:
                    faithfulness_obj = check_answer_faithfulness(final_answer_obj, relevant_chunks, llm_for_json_outputs)
                elif final_answer_obj:
                     pass 
                else:
                    print("CRITICAL ERROR: Answer generation returned None.")
                    final_answer_obj = StructuredAnswer(comprehensive_answer="Critical error in answer generation.", detailed_references=[], confidence_score=0.0, issues_or_missing_info="System error during answer generation.")
            
            st.session_state.last_debug_messages = current_query_debug_messages
            
            assistant_message_content = final_answer_obj.comprehensive_answer if final_answer_obj else "Sorry, I encountered an issue processing your request."
            st.session_state.chat_history.append({
                "role": "assistant", "content": assistant_message_content,
                "structured_answer_obj": final_answer_obj, "faithfulness_check_obj": faithfulness_obj
            })
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("This chatbot uses local Ollama models for enhanced privacy and control over your data.")
st.sidebar.caption("Ensure your Ollama server is running and all required models are pulled.")

