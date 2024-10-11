import streamlit as st
from streamlit_extras.stylable_container import stylable_container
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

# Stylable Container to CSS styles and tricky columns to align button to right
_, col = st.columns([10, 1])
with col:
    with stylable_container(
            key="send_button",
            css_styles="""
                button {
                    background-color: blue;
                    color: white;
                    border-radius: 20px;
                    display: flex;
                    justify-content: flex-end;
                }
                """,
        ):
            st.button("Send")
