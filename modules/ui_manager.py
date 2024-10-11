import streamlit as st
import os

class UIManager:
    def __init__(self, app_path, title, description):
        # self.data_manager = data_manager
        self.app_path = app_path
        self.title = title
        self.description = description
        
        self.logo_path = os.path.join(self.app_path, r'src\images\logo_holmes_enhanced_transp.png')
        self.chat_page = None
        self.about_page = None

        
    def render_ui(self):
        self._set_ui_settings()
        self._show_title(self.title)
        self._show_description(self.description)
        self._show_sidebar()

    def _set_ui_settings(self):
        pass
        # Remove top part of application inutile header

        # Setting sidebar background White
        st.markdown("""
        <style>
            [data-testid=stHeader] {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True)

        # st.markdown('''
        # <style>
        #     [data-testid=stHeader] {
        #         background-color: #ff000050;
        #     }
        # </style>
        # ''', unsafe_allow_html=True)

    def _show_title(self, title):
        st.title(title)

    def _show_description(self, description):
        st.write(description)

    #     with st.Page():
    #         st.subheader('Ask me anything...')
    #         # Question
    #         message = st.text_area(label='message', label_visibility='hidden', placeholder='Ex: "O que significa a sigla EPEP?"', height=150)




    # def _show_main_screen(self):
    #     st.subheader('Ask me anything...')
    #     # Question
    #     message = st.text_area(label='message', label_visibility='hidden', placeholder='Ex: "O que significa a sigla EPEP?"', height=150)






    def _show_sidebar(self):
        with st.sidebar:
            st.image(self.logo_path, width=270)
            st.divider()
