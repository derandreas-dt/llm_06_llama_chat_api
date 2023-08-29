""" This is a more robust chat LLM, where the code is more reusable.
It uses a wrapper to create systematically the chat history.

It provides a system prompt and the question/answer/question/answer logic.

To have a system prompt, the first INST should include the `<<SYS>><</SYS>>` tags.
```
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Never lie and if you are not sure
about something, tell it to the user.
<</SYS>>

<user question 1> [/INST]
```

The output is colored in an xterm:
```
      SYSTEM> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        USER> hello, my name is andreas. What is the capitol of Germany
   ASSISTANT> Hello Andreas! The capital of Germany is Berlin. It is located in the southeastern part of the country and is known for its rich history, cultural landmarks, and vibrant nightlife. I hope that helps! Let me know if you have any other questions.
        USER> thank you, can you tell me capitol of france?
   ASSISTANT> Of course, Andreas! The capital of France is Paris. It is located in the northern part of the country and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. I hope that helps! Let me know if you have any other questions.
        USER> Does it is Brasil or Brazil?
   ASSISTANT> Great catch! The correct answer is "Brazil." Thank you for pointing that out. Here's a corrected version of the previous response:

The capital of Brazil is Brasília. It is located in the central-western part of the country and was built as a new capital city in the 1950s to replace Rio de Janeiro as the country's seat of government. Brasilia is known for its modernist architecture and cultural attractions, such as the National Assembly, the Cathedral, and the Palace of Justice. I hope that helps! Let me know if you have any other questions.
        USER> Can you tell me which country has more people?
   ASSISTANT> Of course! Brazil has a larger population than Germany. According to the most recent estimates, Brazil has a population of approximately 212 million people, while Germany has a population of around 83 million people. I hope that helps! Let me know if you have any other questions.
        USER> Tell me what I said my name is in the first questio
   ASSISTANT> Of course! In your first question, you mentioned that your name is Andreas.
```

This is based on:
`https://github.com/viniciusarruda/llama-cpp-chat-completion-wrapper`
"""
from langchain.llms import LlamaCpp
from ChatPromptWrapper import LlamaChatWrapper
from utils import print_chat_colored

# PATH where the model is on your device
MODEL_PATH = "/home/andreas/development/ai/models/llama-2-7b-chat.ggmlv3.q4_0.bin"

# init the LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048, # context size
    n_gpu_layers=10, # layers to shift to qpu if possible
)

wrap = LlamaChatWrapper(llm)
sess = wrap.new_session()

# create a fake dialog
sess("hello, my name is andreas. What is the capitol of Germany")
sess("thank you, can you tell me capitol of france?")
sess("Does it is Brasil or Brazil?")
sess("Can you tell me which country has more people?")
sess("Tell me what I said my name is in the first questio")

# print it colored
print_chat_colored(sess.get_messages())
