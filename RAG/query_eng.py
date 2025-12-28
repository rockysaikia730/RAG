import streamlit as st
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate
import asyncio

@st.cache_resource
def load_reranker():
    """Loads the reranker model once and caches it."""
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", 
        top_n=5
    )

@st.cache_resource
def load_meta_replacer():
    """Initializes the metadata replacer once."""
    return MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

reranker = load_reranker()
meta_replacer = load_meta_replacer()


custom_prompt = PromptTemplate(template="""
You are a precise and factual assistant. Your task is to answer the user's question based *only* on the provided context.

Follow these rules:
1. Base the answer strictly on the context. Do not guess, infer, or add outside information.
2. If the question asks for more than one item (e.g., list, multiple names, all, etc.), extract *every* relevant item found in the context.
3. When listing multiple items, return them as a clean bullet list, one item per line, with no commentary.
4. Quote names, dates, values, and titles exactly as they appear in the context.
5. If the context does not contain the answer or the requested items are incomplete, respond with:
   "Based on the provided information, I could not find an answer to that question."

---------------------
Context:
{context_str}
---------------------

Question:
{query_str}

Answer:
""")

response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=custom_prompt
)

async def rag_pipeline(user_query, llm_model):
    # These objects (reranker, etc.) are available 
    # from the main script's scope.
    fusion_retriever = st.session_state.fusion_retriever

    nodes = await fusion_retriever.aretrieve(user_query)

    replaced_nodes = meta_replacer.postprocess_nodes(
        nodes=nodes
    )
    reranked_nodes = reranker.postprocess_nodes(
        nodes=replaced_nodes, 
        query_str=user_query
    )
    
    # Note: .synthesize() is often not async. 
    # If it is, you should await it.
    final_response = response_synthesizer.synthesize(
        query=user_query,   
        nodes=reranked_nodes 
    )
    
    return final_response