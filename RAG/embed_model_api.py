import os
from dotenv import load_dotenv
import sys
from exception import customexception
from logger import logging

import streamlit as st

load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

@st.cache_resource
def load_embed_model(model_name="mxbai-embed-large:latest"):
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding 
        if model_name == "mxbai-embed-large:latest":
            embed_model = OllamaEmbedding(model_name="mxbai-embed-large:latest")
        else:
            embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")
        return embed_model
    except Exception as e:
        raise customexception(e,sys)
