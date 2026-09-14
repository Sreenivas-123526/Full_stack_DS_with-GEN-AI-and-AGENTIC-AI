from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()

app = FastAPI(title="APPLICATION created by PALUTLA")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 200


class ChatResponse(BaseModel):
    response: str
    model: str
    input_tokens: int
    output_tokens: int


@app.get("/")
def hello():
    return {"message": "Hi how can I help you"}


@app.post("/chatbot")
def chat_stream(req: ChatRequest):

    def generate_res():

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": req.message
                }
            ],
            max_tokens=req.max_tokens,
            stream=True
        )

        for chunk in stream:
            text = chunk.choices[0].delta.content

            if text:
                yield f"data: {text}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_res(),
        media_type="text/event-stream"
    )