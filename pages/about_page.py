import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

add_vertical_space()

st.markdown(
'''
**H.O.L.M.E.S.** is the acronym for **H**istorical **O**bservation and **L**earning **M**aterials **E**ngineering **S**ystem, a **ChatBot** AI system based on LLM models and **business knowledge bases**, configuring a **RAG** (Retrieval-Augmented Generation) architecture that combines the best **generative Artificial Intelligence models to deal with natural language processing** and answer questions based on a specific context.

The system is fed with several data sources in different formats: structured and unstructured, and uses a similarity search to feed the LLM model with relevant content related to the user's demand.

**The system's performance varies mainly according to the efficiency of the models used in the embedding and conversation structures**, and different tests can be carried out with different models, taking into account the installed computing capacity, the data governance concern and the system's objective.

'''
, unsafe_allow_html=True)

add_vertical_space(3)

st.write('For additional questions, reach me at:')
st.markdown('**paulo.araujo@embraer.com.br**')
st.markdown('**Paulo Roberto de Sá Araújo**')
add_vertical_space(4)