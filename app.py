import streamlit as st
from transformers import pipeline
import pandas as pd

class ChatBot:
    def __init__(self, name, description, conversation):
        self.name = name
        self.description = description
        self.conversation = conversation