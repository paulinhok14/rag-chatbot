import streamlit as st
import os
from modules import UIManager
#, DataManager



class AppMain:
    def __init__(self):
        # Initialize app state]
        self.app_path = os.path.dirname(__file__)
        self.title = "H:blue[.]O:blue[.]L:blue[.]M:blue[.]E:blue[.]S:blue[.] :male-detective:"
        self.description = 'Materials Solution AI Assistant'
        self.knowledge_base_path = os.path.join(os.path.dirname(__file__), 'association_rules.xlsx')
        self.ui_manager = UIManager(self.app_path, self.title, self.description)
        # self.data_manager = DataManager(self.knowledge_base_path)
        

    def run(self):
        # Set the page configuration
        st.set_page_config(
            page_title='H.O.L.M.E.S.', 
            page_icon=":male-detective:",
            layout='wide',
            initial_sidebar_state ='expanded'
        )
        # Load data
        # self.data_manager.load_data()
        # self.data_manager.load_complementary_data()

        # Call methods to render interface parts
        self.ui_manager.run_ui()



if __name__ == "__main__":
    app = AppMain()
    app.run()
