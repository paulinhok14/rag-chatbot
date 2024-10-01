import streamlit as st
import os
from transformers import pipeline
from dotenv import load_dotenv
import pandas as pd
from docx import Document
import ollama
from groq import Groq
import sys

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Bottomline: AI Powered.

# Load environment variables
load_dotenv()

file_path = os.path.join(os.path.dirname(__file__), r'src\databases\Bizuario Geral.docx')

def retrieve_info(query):
    '''
    Function that searchs for similarity in text database (vector store) starting from query, stated by user.
    Retrieves k Documents.
    '''
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]


# @st.cache_resource
def load_model():
    '''
    Load the Language model
    '''
    return pipeline("question-answering", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")

# Prompt
template = '''
Você é um assistente virtual de uma área de materiais chamada Spare Parts Planning.
Sua função será responder à questões genéricas feitas pelos colaboradores da área.
Vou lhe passar um documento com diversas informações relevantes, tais como significado de siglas, telefones úteis, explicação de processos, etc, para que você use como referência para responder ao questionamento do usuário.

Siga todas as regras abaixo:
1/ Você deve buscar se comportar de maneira sempre cordial e solícita para atender aos questionamentos dos usuários.

2/ Algumas linhas do documento fornecido podem conter informações irrelevantes. Preste atenção ao conteúdo útil da mensagem.

3/ Existem informações pessoais dos colaboradores no documento, tais como número de telefone, evite passá-las sob quaisquer circunstâncias.
A única informação que pode ser passada relacionada aos colaboradores é o respectivo login.

4/ Em hipótese alguma envolva-se em discussões de cunho pessoal, sobre tecer opiniões sobre um colaborador ou outro. Caso a pergunta seja neste sentido, recuse-se gentilmente a opinar e ofereça ao usuário ajuda nas questões relevantes.

Aqui está uma pergunta recebida de um usuário.
{question}

Aqui está o documento com as informações relevantes mencionadas.
Esse arquivo servirá de base para que você compreenda nosso contexto de negócio, organização e nossas siglas mais comumente utilizadas.
{bizuario_document}

Escreva a melhor resposta que atende ao questionamento do usuário:
'''
    
# prompt = PromptTemplate(
#     input_variables=['question', 'bizuario_document'],
#     template=template
# )


def generate_response(question):
    '''
    Function that takes a question made by user and returns an answer
    '''
    relevant_info = retrieve_info(question)
    # Feeding LLM
    response = chain.run(question=question, bizuario_document=bizuario_document)
    return response

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    embeddings_model = SentenceTransformer('jinaai/jina-embeddings-v3')
    return embeddings_model



# Querying
#retrieve_info(query='O que significa a sigla EPEP?')

# Instancing LLM model
#llm = load_model()

# Chain
#chain = LLMChain(llm=llm, prompt=prompt)

def main():
    # # Page config
    # st.set_page_config(
    #     page_title="Stella.AI", page_icon=':robot_face:'
    # )

    # st.header('Stella.AI')

    # question = st.text_area('Ask me anything related to Spare Parts Planning')

    # # Conditional exhibiting components
    # if question:
    #     st.write("Fetching response... :robot_face:")

    #     result = generate_response(question)

    #     st.info(result)

    


    # Estrutura:

    '''
    1- Gerar Base de Conhecimento. Ler word e usar modelo de Embedding (FastEmbed?) para jogar para a Vector Store db
    2- Criar um Retriever
    3- Instanciar um modelo LLM (modelo= llama3-8b preferencialmente? E uma API (Groq?))
    # llm = ChatGroq(temperature=0, model_name='llama-3.1-8b-instant') -> GROQ API tem 30 milhões de tokens (muito) gratuito sem se preocupar com infra agora, plataf cloud.
    4- Determinar prompt: system message
    5- Criar um chain com LangChain conectando o LLM com o Retriever e o Prompt (chain_type='stuff')
    
    
    '''

    # 1- Create Knowledge Base

    # Reading doc
    loader = Docx2txtLoader(file_path)
    bizuario_doc = loader.load()
    # Splitting text into chunks
    chunks = split_documents(bizuario_doc)
    # Embedding model to Vector Store transforming
    embeddings = get_embedding_function()
    # Vector Store
    #db = FAISS.from_documents(bizuario_doc, embeddings)
    print(chunks[0])


    # # Groq API Client instancing
    # client = Groq(
    #     api_key=os.getenv('GROQ_API_KEY'),
    # )












if __name__ == "__main__":
    main()