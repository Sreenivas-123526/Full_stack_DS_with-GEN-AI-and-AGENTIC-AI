from dotenv import load_dotenv

load_dotenv()

import pathlib
import textwrap
from google import genai
from IPython.display import display
from IPython.display import Markdown

import os

import streamlit as st

def to_markdown(text):
    text=text.replace('.','*')
    return Markdown(textwrap.indent(text,'>',predicate=lambda _:True))

def Generate_response(question):
    Client=genai.Client(
        api_key=os.environ['GEMINI_API_KEY']
    )

    try:
        response=Client.models.generate_content(
            model='models/gemini-3-flash-preview',
            contents=question
        )

        return response.text
    except Exception as e:
        print('Actual error',e)


st.set_page_config(page_title='GEMINI APPLICATION')

st.title("Gemini CHATBOT created by PALUTLA")

input=st.text_input("input :",key="input")


if st.button("submit"):
    with st.spinner('Thinkinggggggggggggg'):
        response=Generate_response(input)
        st.subheader("responde is")
        st.write(response)
