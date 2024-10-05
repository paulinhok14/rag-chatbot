from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from credentials import DBNAME, USER, PASSWORD, HOST, PORT

# warnings.filterwarnings('ignore')

file_path = os.path.join(os.path.dirname(__file__), r'src\databases\Bizuario Geral.docx')

loader = AzureAIDocumentIntelligenceLoader(file_path, encoding='utf-8', api_model="prebuilt-layout")
documents = loader.load()
print(documents)