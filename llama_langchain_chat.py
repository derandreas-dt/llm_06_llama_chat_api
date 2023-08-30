""" This is a more evolved ai chat system
which uses the langchain modules where possible.

Keeps track of the history with ChatMessageHistory
"""
from langchain.llms import LlamaCpp
from ChatPromptWrapper import LlamaChatWrapper
from utils import print_memory_colored

# PATH where the model is on your device
MODEL_PATH = "/home/andreas/development/ai/models/llama-2-7b-chat.ggmlv3.q4_0.bin"

# init the LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048, # context size
    n_gpu_layers=10, # layers to shift to qpu if possible
)

wrap = LlamaChatWrapper(llm)
conversation_id, sess = wrap.new_session()

# create a fake dialog
sess("hello, my name is andreas. What is the capitol of Germany")
sess("thank you, can you tell me capitol of france?")
sess("Does it is Brasil or Brazil?")
sess("Can you tell me in which country live the most people?")
sess("Tell me what I said my name is from the first question")

print_memory_colored(sess.memory)
