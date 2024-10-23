import streamlit as st
import os
import numpy as np
import time

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain.prompts import PromptTemplate
from operator import itemgetter

# To deal with error #15 initializing FAISS parallelism: 'Initializing libomp140.x86_64.dll, but found libiomp5md.dll'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Model instancing
MODEL = 'llama3.2'
EMBEDDINGS_MODEL = 'nomic-embed-text'
# MODEL = 'llama2:7b-chat'
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

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
Seu nome é H.O.L.M.E.S.
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

def generate_response(question):
    '''
    Function that takes a question, retrieves relevant documents and generates a response
    '''

    with st.spinner('Investigating for answers... 🔎'):
        print('Question: ', question)
        print('\n')
        relevant_content = retriever.invoke(question)
        for i, content in enumerate(relevant_content):
            print(f'{i+1}: \n', content.page_content)
            print('\n')
        print('\n')
        #print('Generated prompt: \n\n', prompt.format(question=question, context=relevant_content))
        
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
    # return answer
    print('answer:\n', answer)
    # Generating write stream respecting line breaks
    # for word in answer.split():
    #     yield word + " "
    #     time.sleep(0.05)
    for line in answer.split('\n'):
        for word in line.split():
            yield word + ' '
            time.sleep(0.05)
        yield '\n'  # Adds a break after each line
        time.sleep(0.1)

def generate_fake_response(question):
    empty_space = st.empty()
    with empty_space.container():

        with st.status('Generating response...', expanded=False) as status:

            time.sleep(5)
            fake_answer = 'Resposta Gerada Após o Processamento do Modelo'
            status.update(
                label='Response Generated!', state='complete', expanded=None
            )
    empty_space.empty()

    for word in fake_answer.split():
        yield word + " "
        time.sleep(0.05)

# Chat Components

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(name=message["role"], avatar=message['avatar']):
        st.markdown(message["content"])

# def capitalize_first_word():
#     '''
#     This is a callback function that checks if the word being written on Chat st.text_area() is the first word, and if so, capitalizes it.
#     '''
#     text = st.session_state['question-input']

#     if text and text[0].islower():
#         # Updates text by session_state with the first word capitalized
#         st.session_state['question-input'] = text[0].upper() + text[1:]


# Reacting to user input (question)
if question := st.chat_input(placeholder='Ex: "O que significa a sigla APU?"', max_chars=2000):
    # Display user message in chat_message container
    st.chat_message(name='user', avatar='👨‍💼').markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': question, 'avatar': '👨‍💼'})

    # Display H.O.L.M.E.S. response in chat_message container, streaming words
    # st.chat_message(name='assistant', avatar='🕵️‍♂️').markdown(result)
    with st.chat_message(name='assistant', avatar='🕵️‍♂️'):
        result = st.write_stream(generate_response(question))
    # Add H.O.L.M.E.S. response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': result, 'avatar': '🕵️‍♂️'})
    

# Answer space
if question:
    answer_space = st.empty()
    # Generate Answer running the chain
    # result = generate_response(question)
    # result = generate_fake_response(question)
    # Set answer as "ready" and shows final answer
    # st.session_state["response_ready"] = True
    #answer_space.info(result)

    # Mock questions
    questions = [
        'O que é o AHEAD?',
        'O que significa MTBF?',
        'Onde ficam os KPIs da área?',
        'Quais são as aeronaves Embraer?',
        'O que significa APU?',
        'Por que eu devo evitar fazer críticas em público?',
        'O que é a matriz GUT?',
        'Qual é o email do Warehouse?',
        'O que acha do Diego Sodre?'
        'O que faz a transação ZLOLMM006?'
        'Quais são os processos P3E?',
        'Quais são algumas das regras para uma boa convivência com os  colegas?'
    ]