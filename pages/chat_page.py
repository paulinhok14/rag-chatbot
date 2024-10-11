import streamlit as st
import os

# Initialize Page state
description = 'Materials Solution AI Assistant'
knowledge_base_path = os.path.join(os.path.dirname(__file__), 'association_rules.xlsx')

# Page components
st.write(description)
st.subheader('Ask me anything...')

# Chat components

# Question
message = st.text_area(label='message', label_visibility='hidden', placeholder='Ex: "O que significa a sigla EPEP?"', height=150)


