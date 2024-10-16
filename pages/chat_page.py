import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import numpy as np
import time
import threading

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch # Get a better solution instead of saving in memory. At first, works.
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain.prompts import PromptTemplate
from operator import itemgetter

# To deal with error #15 initializing FAISS parallelism: 'Initializing libomp140.x86_64.dll, but found libiomp5md.dll'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Model instancing
MODEL = 'llama3.2'
# MODEL = 'llama2:7b-chat'
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# FAISS index path
faiss_index_path = os.path.join(os.getcwd(), 'src', 'databases', 'faiss_index')
# Knowledge Base path
bizuario_doc_path = os.path.join(os.getcwd(), r'src\databases\Bizuario Geral.docx')

def split_documents(documents):
    '''
    Splits documents in smaller chunks of text, in order to pass to LLM more specific context
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# Function to load FAISS vectorstore database if it exists, or create a new one based on Bizuario doc
def load_or_create_faiss_index():
    if os.path.exists(faiss_index_path):
        # Load existing FAISS index
        return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True) # Attention to allow_dangerous_deserialization prm, only trusted sources.
    else:
        # Create Knowledge Base with FAISS
        loader = Docx2txtLoader(bizuario_doc_path)
        docs = loader.load_and_split()
        # Splitting text into chunks (each document is a page, so chunks has to be smaller in order to provide a more specific context to LLM model)
        chunks = split_documents(docs)
        # Generating embeddings for all documents
        chunks_embeddings = [embeddings.embed_query(chunk.page_content) for chunk in chunks]

        # Initializing FAISS index
        dimension = len(chunks_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        # Adding embeddings to index
        index.add(np.array(chunks_embeddings))
        # Creating docstore and IDs mapping
        docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(chunks)})
        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}
        # Creating FAISS vectorstore
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        # Saving vectorstore in disk
        vectorstore.save_local(faiss_index_path)

        return vectorstore


# Starting vectorstore (Loading Knowledge Base data)
vectorstore = load_or_create_faiss_index()
# Retriever and respective parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "lambda_mult": 0.5},
    # search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
    # The Retriever can perform “similarity” (default), “mmr”, or “similarity_score_threshold”. Test them all.

)

# Initialize Page state
description = 'Materials Solution AI Assistant'
knowledge_base_path = os.path.join(os.path.dirname(__file__), 'association_rules.xlsx')

# Page components
st.write(description)
st.subheader('Ask me anything...')

def stream_text(text, delay=0.04):
    '''
    Function that takes a text as a String and returns a stream generator in order to be displayed with st.write_stream()
    '''
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


# Prompt (System Message. Define Tasks, Tone, Behaviour and Safety params)
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

Aqui está o documento com as informações relevantes mencionadas.
Esse arquivo servirá de base para que você compreenda nosso contexto de negócio, organização e nossas siglas mais comumente utilizadas:
{context}

Aqui está uma pergunta recebida de um usuário:
{question}

Escreva a melhor resposta que atende ao questionamento do usuário, de forma precisa e objetiva.
'''

prompt = PromptTemplate.from_template(template)

def show_loading_message(answer_space):
    pass

def generate_response(question):
    print('Question: ', question)
    print('\n')
    relevant_content = retriever.invoke(question)
    for content in relevant_content:
        print(content.page_content) 
    print('Relevant content: \n\n', relevant_content)
    print('\n')
    #print('Generated prompt: \n\n', prompt.format(question=question, context=relevant_content))
    print('\n')
    
    # Creating and invoking Chain    
    chain = (
        {
            'context': itemgetter('question') | retriever, # the context will come from a retriever (relevant docs), given a question
            'question': itemgetter('question')
        }
        | prompt
        | model
    )

    answer = chain.invoke({'question': question})
    return answer

# Chat Components
# def capitalize_first_word():
#     '''
#     This is a callback function that checks if the word being written on Chat st.text_area() is the first word, and if so, capitalizes it.
#     '''
#     text = st.session_state['question-input']

#     if text and text[0].islower():
#         # Updates text by session_state with the first word capitalized
#         st.session_state['question-input'] = text[0].upper() + text[1:]

# Question
question = st.text_area(label='question', 
                        label_visibility='hidden', 
                        placeholder='Ex: "O que significa a sigla APU?"', 
                        height=150, 
                        help='Press Ctrl+Enter to send question',
                        key='question-input'
                        )

# Answer space
if question:
    answer_space = st.empty()


    # Generate Answer running the chain
    result = generate_response(question)
    # Set answer as "ready" and shows final answer
    # st.session_state["response_ready"] = True
    answer_space.info(result)

    # Mock questions
    questions = [
        'O que significa a sigla EPEP?',
        'O que faz a transação ME22N?',
        'Qual transação posso usar para modificar a Ordem de Cliente (OV/SO)?',
        'O que significa APU?',
        'Quais são os processos P3E?',
        'Qual é o ramal do Diego Sodre?',
        'Qual é a chapa do Diego Sodre?',
        'Qual é o ramal do ambulatório?',
        'Quais são algumas das regras para uma boa convivência com os  colegas?',
        'Por que eu devo evitar fazer críticas em público?'
    ]

    # for question in questions:
    #     resposta = generate_response(question)
    #     st.info('Question:\n'+question)
    #     print('\n')
    #     st.info('Answer:\n'+resposta)
    #     print('\n')