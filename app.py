import streamlit as st
import os

def main():
    # Initialize app state
    app_path = os.path.dirname(__file__)
    title = "H:blue[.]O:blue[.]L:blue[.]M:blue[.]E:blue[.]S:blue[.] :male-detective:"
    logo_path = os.path.join(app_path, r'src\images\logo_holmes_enhanced_transp.png')

    # Initializing Page configs
    st.set_page_config(
        page_title='H.O.L.M.E.S.', 
        page_icon="🕵️‍♂️",
        layout='wide',
        initial_sidebar_state ='expanded'
        )
    st.title(title)
   
    # Chat Page
    chat_page = st.Page(
        page='pages/chat_page.py',
        title='Chat',
        icon='💬',
        url_path='pages/chat_page.py',
    )
    # About Page
    about_page = st.Page(
        page='pages/about_page.py',
        title='About',
        icon='📝',
        url_path='pages/about_page.py'
    )

    # Sidebar
    with st.sidebar:
        # Logo
        st.image(logo_path, width=270)
        st.divider()

        # Page Linking
        st.page_link(page=chat_page, label="Chat", icon="💬")
        st.page_link(page=about_page, label="About", icon="📝")

    # Navigation
    nav = st.navigation(pages=[chat_page, about_page], position='hidden')
    nav.run()


if __name__ == "__main__":
   main()