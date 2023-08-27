""" Create a simple LLM with local model without any fancy stuff
TODO: more explenation

This will init a LLM with the local model, which can be something like
the quantized model `llama-2-7b-chat.ggmlv3.q4_0.bin`.
Please note that the GGML quantization format is EOL since August 2023
and replaced by the GGUF format. Download the respective model or
stick with `llama-cpp-python=0.1.78`.

A template is setup, which is a system prompt to tell the LLM what it is.
This is here very basic and nothing special. This is transformed into
a PromptTemplate for reusability and passed with the LLM into a LLMChain.

This chain can be run, by using `llm_chain.run("question as string")`

The result is then printed and could look like this:

```
    The result of the AI is: ==>

    Germany: The capital of Germany is Berlin.

    France: The capital of France is Paris.

    Brasil (not a country): Brasil is not a country, it's the Portuguese name for Brazil. The capital of Brazil is Brasília.
    <== end of result
```

This is it for the first step.
"""

from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp

# PATH where the model is on your device
MODEL_PATH = "/home/andreas/development/ai/models/llama-2-7b-chat.ggmlv3.q4_0.bin"

# define a system template for a prompt
# NOTE the system template prompt is wrapper for the actual user question

system_template = """
Question: {question}
Answer: Let's think step by step
"""

prompt = PromptTemplate(template=system_template, input_variables=["question"])

# init the LLM

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048, # context size
    n_gpu_layers=10, # layers to shift to qpu if possible
)

# create the llm chain with prompt, lmm
llm_chain = LLMChain(llm=llm, prompt=prompt)

# test the first question and simple print the llm result
ai_result = llm_chain.run("What is the capitol of Germany, France and Brasil?")
print(f"The result of the AI is: ==>\n{ai_result}\n<== end of result")

