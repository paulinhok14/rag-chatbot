import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv
import pandas as pd

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader


def main():

    # Load environment variables
    load_dotenv()

    # Setup
    databases_path = st.secrets["databases_path"]

    @st.cache_resource
    def load_model():
        '''
        Load the Language model
        '''
        return pipeline("question-answering", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")
    
    # Carregar a base de dados Excel
    @st.cache_data
    def load_databases(file_path):

        # TODO: IMPLEMENTAR LOADER PARA LER O BIZUARIO EM WORD E RESPONDER A UM QUESTIONAMENTO, TESTE. USAR LLM GRATUITA HUGGING FACE
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo Excel: {e}")
            return None

if __name__ == "__main__":
    main()