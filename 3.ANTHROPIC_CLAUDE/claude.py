from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from anthropic import Anthropic

import os

from dotenv import load_dotenv

load_dotenv()

app=FastAPI(title="Claude model with FastAPI")
client=Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

#request |response 

class ChatRequest(BaseModel): #ChatRequest = what the client sends to FastAPI
    message:str
    max_tokens:int=1024

class ChatResponse(BaseModel): #ChatResponse = what FastAPI sends back to the client
    response:str
    model:str
    input_tokens:int
    output_tokens:int

@app.get('/')

def root():
    return {"status":'OK',"message":'Welcome to claue model'}

@app.post('/chatbot')

def chat_stream(req:ChatRequest):
    from fastapi.responses import StreamingResponse

    def generate():
        with client.messages.stream(
            model='claude-sonnet-4-5',
            messages=[{"role":"user","content":req.message}],
            max_tokens=req.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                
                yield f'data:{text}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(),media_type="text/event=stream")




