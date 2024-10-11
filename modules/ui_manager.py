import streamlit as st
import os

class UIManager:
    def __init__(self, app_path):
        # self.data_manager = data_manager
        self.app_path = app_path
        self.logo_path = os.path.join(self.app_path, r'src\images\logo_holmes_enhanced.png')

    def show_title(self, title):
        st.title(title)

    def show_description(self, description):
        st.write(description)

    def show_sidebar(self):
        with st.sidebar:
            #st.image()
            st.button('teste')