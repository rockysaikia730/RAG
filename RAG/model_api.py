import os
from dotenv import load_dotenv
import sys
from exception import customexception
from logger import logging


import streamlit as st

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_model(model_name):
    try:
        
        if model_name == 'Gemini':
            from llama_index.llms.gemini import Gemini
            import google.generativeai as genai
            model = Gemini(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
        else:
            from llama_index.llms.groq import Groq
            model = Groq(model=model_name, api_key=GROQ_API_KEY)
        
        return model
    except Exception as e:
        raise customexception(e,sys)

