""" Chat system which mimics the chat.openai.com services

Contains
    * an api server that handles the api calls with llm backend
    * an http static service to serve the html / css / js
    * a simple website to have a ui

This is terriable code at the frontend side, but wanted to have
a very simple pure js version, without some nasty frontend framework.
With something like vue/react this would be much more easy, but
didn't want that pipeline and it should be working with fastapi directly.

The frontend allow some features like multiple chats.
All chats are stored only on the client side (localstorage) and
by deleting it, its all gone.

Also on the backend side there is only a memory store. Restarting
the app prunes that memory and the messages in the client are not
sync with the backend anymore.

"""
import logging

from uuid import uuid4
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from langchain.llms import LlamaCpp
from ChatPromptWrapper import LlamaChatWrapper

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# PATH where the model is on your device
MODEL_PATH = "/home/andreas/development/ai/models/llama-2-7b-chat.ggmlv3.q4_0.bin"

# init the LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048, # context size
    n_gpu_layers=10, # layers to shift to qpu if possible
)

llama = LlamaChatWrapper(llm)

app = FastAPI(
    title="FastAPI Server for LLM system",
    version="0.0.1",
)


class ChatRequest(BaseModel):
    conversationId: str
    query: str

@app.get('/api/ping')
async def ping():
    """ Simple ping route, not used atm

    Returns
    -------
    dict
    """
    return {"message": "pong"}

@app.post('/api/query')
async def query(data: ChatRequest):
    """ Actual query route to ask the LLm for a response

    Parameters
    ----------
    data: ChatRequest

    Returns
    -------
    dict
    """
    if data.conversationId.startswith('prepare-dummy-'):
        conversation_id = uuid4()
    else:
        conversation_id = data.conversationId

    session = llama.new_session(conversation_id)

    res = session(data.query)

    return {
        "conversationId": conversation_id,
        "message": res,
    }


app.mount("/", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)
