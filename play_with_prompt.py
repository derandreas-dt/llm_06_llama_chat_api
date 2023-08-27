""" Play with the simple prompt, extend it and let the Ai behave differently

This modifies the prompt template by adding a new variable, which
is the in the llm_chain.run call set with a new instruction

The input of the llm_chain prompt is now (2 variables, `question` and `new_variable`)

```What is the capitol of Germany, France and Brasil? ```
and
```Answer everything with I-LIKE-BANANAS```

The result is printed and could look like this:

```
The result of the AI is: ==>
I LIKE BANANAS! *smiles* The capitol of Germany is... I LIKE BANANAS! *nods* It's Berlin! *excitedly* I LIKE BANANAS!
The capitol of France is... I LIKE BANANAS! *thinks* Hmm, let me see... it's Paris! *smiling* I LIKE BANANAS!
And the capitol of Brasil? *thoughtful expression* Oh, that's a tough one! Let me think... *pauses* I LIKE BANANAS! *excitedly* It's Brasília! *nods* I LIKE BANANAS!
<== end of result
```

Whoops, expressions and a differnt style, but I LIKE BANANAS!
"""

from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp

# PATH where the model is on your device
MODEL_PATH = "/home/andreas/development/ai/models/llama-2-7b-chat.ggmlv3.q4_0.bin"

# define a system template for a prompt
# NOTE the system template prompt is wrapper for the actual user question

system_template = """
Question: {question} {new_variable}
Answer: Let's think step by step
"""

prompt = PromptTemplate(template=system_template, input_variables=["question", "new_variable"])

# init the LLM

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048, # context size
    n_gpu_layers=10, # layers to shift to qpu if possible
)

# create the llm chain with prompt, lmm
llm_chain = LLMChain(llm=llm, prompt=prompt)

# test the first question and simple print the llm result
question = "What is the capitol of Germany, France and Brasil?"
new_variable = "Answer everything with I-LIKE-BANANAS"
ai_result = llm_chain.run(question=question, new_variable=new_variable)
print(f"The result of the AI is: ==>\n{ai_result}\n<== end of result")

