import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv
import pandas as pd
from docx import Document

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader


def main():

    # Load environment variables
    load_dotenv()

    file_path = r'./src/databases/Bizuario Geral.docx'

    # Reading "Bizuario" Word document
    # @st.cache_data()
    def read_word_document(file_path):
        doc = Document(file_path)
        texto = []
        for paragrafo in doc.paragraphs:
            texto.append(paragrafo.text)
        return "\n".join(texto)

    # Calling read document function
    bizuario_document = read_word_document(file_path)


    # Embeddings object instancing
    embeddings = OpenAIEmbeddings()
    # Vector Store
    db = FAISS.from_documents(bizuario_document, embeddings)

    print('chegou aqui')
    print(bizuario_document)

    @st.cache_resource
    def load_model():
        '''
        Load the Language model
        '''
        return pipeline("question-answering", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")
    

        

if __name__ == "__main__":
    main()