import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import time

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch # Get a better solution instead of saving in memory. At first, works.
from langchain.prompts import PromptTemplate
from operator import itemgetter

# Model instancing
MODEL = 'llama3.2'
# MODEL = 'llama2:7b-chat'
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Knowledge Base
bizuario_doc_path = os.path.join(os.getcwd(), r'src\databases\Bizuario Geral.docx')
# Reading file as documents
loader = Docx2txtLoader(bizuario_doc_path)
bizuario_doc = loader.load_and_split()
# Starting VectorStore
vectorstore = DocArrayInMemorySearch.from_documents(bizuario_doc, embedding=embeddings)
# Retriever
retriever = vectorstore.as_retriever()

# Initialize Page state
description = 'Materials Solution AI Assistant'
knowledge_base_path = os.path.join(os.path.dirname(__file__), 'association_rules.xlsx')

# Page components
st.write(description)
st.subheader('Ask me anything...')

def stream_text(text):
    '''
    Function that takes a text as a String and returns a stream generator in order to be displayed with st.write_stream()
    '''
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)


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

Aqui está uma pergunta recebida de um usuário:
{question}

Aqui está o documento com as informações relevantes mencionadas.
Esse arquivo servirá de base para que você compreenda nosso contexto de negócio, organização e nossas siglas mais comumente utilizadas:
{context}

Escreva a melhor resposta que atende ao questionamento do usuário:
'''

prompt = PromptTemplate.from_template(template)


def generate_response(question):
    print('Question: ', question)
    print('\n')
    relevant_content = retriever.invoke(question)
    print('Relevant content: \n\n', relevant_content)
    print('\n')
    print('Generated prompt: \n\n', prompt.format(question=question, context=relevant_content))
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


# Question
question = st.text_area(label='message', label_visibility='hidden', placeholder='Ex: "O que significa a sigla EPEP?"', height=150, help='Press Ctrl+Enter to send question')

# Chat components
if question:
      answer_space = st.empty()
    #   while True: # Create condition in which the response is already generated so the loop should stop
    #     answer_space.write_stream(stream_text("Investigating for answers. Please wait... :male-detective:"))
    #     time.sleep(2)

      result = generate_response(question)

      st.info(result)
