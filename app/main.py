from fastapi import FastAPI
from app.chatbot import generate_response

app = FastAPI()


@app.get("/")
def home():
    return {"message": "MindSpark Chatbot API running"}


@app.post("/chat")
def chat(message: str):
    reply = generate_response(message)
    return {"response": reply}
