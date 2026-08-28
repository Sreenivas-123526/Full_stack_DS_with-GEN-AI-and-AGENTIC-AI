from dotenv import load_dotenv
import os
from PIL import Image
from google import genai
import streamlit as st

load_dotenv()

api_key=os.getenv('GEMINI_API_KEY')

Client=genai.Client(
    api_key=api_key
)

def get_response(prompt,image):
    response=Client.models.generate_content(
        model='models/gemini-3-flash-preview',
        contents=[prompt,image]
    )

    return response.text


st.set_page_config(page_title='GEMINI APP')

st.header('Application')

prompt=st.text_input(
    "input prompt",key=input
)


uploaded_file=st.file_uploader(
    "choose an image",
    type=["jpg","jpeg","png"]
)


image=None

if uploaded_file is not None:
    image=Image.open(uploaded_file)

    st.image(image,caption="uploaded image",use_container_width=True)


if st.button('tell me about the image'):
    if image is None:
        st.warning("please upload image first")

    else:
        response=get_response(
            prompt,
            image
        )

    st.subheader('The response is')

    st.write(response)
