from exception import customexception
import sys
import sys

import streamlit as st
print(sys.executable)
import RAG.embed_model_api as embed_model_api
import RAG.model_api as model_api

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever


import nest_asyncio
nest_asyncio.apply()

def excluded_metadata(exclude_meta):
    exclude_list = list(exclude_meta)
    if "file_name" in exclude_list:
        exclude_list.remove("file_name")
    if "file_path" in exclude_list:
        exclude_list.remove("file_path")

    exclude_list.extend(["window", "original_sentence","hash_code"])
    return exclude_list

@st.cache_resource
def set_config_indexing(my_exclude_keys, embed_model='mxbai-embed-large:latest'):
    try:
        Settings.embed_model = embed_model_api.load_embed_model(embed_model)
        Settings.llm = st.session_state.model_llm

        Settings.node_parser = SentenceWindowNodeParser(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_sentence",
            excluded_embed_metadata_keys=my_exclude_keys
        )
    except Exception as e:
        raise customexception(e,sys)

def create_or_update_retriever(documents):
    if not documents:
        st.warning("No new documents to process.")
        return

    with st.spinner("Configuring models and parsers..."):
        exclude_metadata = excluded_metadata(documents[0].excluded_llm_metadata_keys)
        set_config_indexing(exclude_metadata)

    all_nodes = st.session_state.get("all_nodes", [])
    vector_index = st.session_state.get("vector_index", None)

    with st.spinner(f"Parsing {len(documents)} new documents..."):
        new_nodes = Settings.node_parser.get_nodes_from_documents(documents, show_progress=True)

    if not new_nodes:
        st.warning("No new nodes were generated from the documents.")
        return

    all_nodes.extend(new_nodes)
    st.session_state.all_nodes = all_nodes

    with st.spinner("Updating retrievers... This may take a moment."):
        if vector_index is None:
            vector_index = VectorStoreIndex(all_nodes, show_progress=True)
            st.session_state.vector_index = vector_index
        else:
            vector_index.insert_nodes(new_nodes)

    import Stemmer
    vretriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes, 
            similarity_top_k=5,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )
    
    st.info("Creating final Retriever...")
    from llama_index.core.retrievers import QueryFusionRetriever
    fusion_retriever = QueryFusionRetriever(retrievers=[vretriever, bm25_retriever],
                                            similarity_top_k=5,     
                                            mode="reciprocal_rerank", 
                                            use_async=True,
                                            verbose=False
                                            )
    
    st.session_state.fusion_retriever = fusion_retriever


