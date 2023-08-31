
from langchain.llms import LlamaCpp
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.schema.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

def _format_message_as_text(message):
    """ Format the message based on its type
        and build the prompt text for the message
        type for the llama2 model

    Parameters
    ----------
    messages: ChatMessage, HumanMessage, AIMessage, SystemMessage

    Returns
    -------
    str
    """
    if isinstance(message, ChatMessage):
        message_text = f"\n\n{message.role.capitalize()}: {message.content}"
    elif isinstance(message, HumanMessage):
        message_text = f"[INST] {message.content} [/INST]"
    elif isinstance(message, AIMessage):
        message_text = f"{message.content}"
    elif isinstance(message, SystemMessage):
        message_text = f"<<SYS>> {message.content} <</SYS>>"
    else:
        raise ValueError(f"Got unknown type {message}")

    return message_text

def _format_messages_as_text(messages):
    """ Formats the joined prompt str based on the
        messages in the ChatMessageHistory store

    Parameters
    ----------
    messages: list
        List of messages from the ChatMessageHistory

    Returns
    -------
    str
    """
    return "\n".join(
        [_format_message_as_text(message) for message in messages]
    )

class LlamaChatSession:
    """ An individual session which is tracked by the wrapper
        This calls the llm model and tracks the messages
        in the session
    """
    llm: LlamaCpp
    memory: ChatMessageHistory

    def __init__(self, llm, memory, system_prompt):
        self.llm = llm
        self.memory = memory

        self.memory.add_message(SystemMessage(content=system_prompt))

    def __call__(self, message):
        self.memory.add_user_message(message)
        res = self.llm(_format_messages_as_text(self.memory.messages))
        self.memory.add_ai_message(res)

        return res

class LlamaChatWrapper:
    """ Reusable wrapper to create a session based
        on the same instance of the LLM.
    """

    def __init__(self, llm):
        """ Init a new Wrapper for the chat

        Parameters
        ----------
        llm: LlamaCpp
        """
        self.llm = llm
        self.memories = {}

    def new_session(self, conversation_id, system_prompt=DEFAULT_SYSTEM_PROMPT):
        """ Creates a new Session by using the wrapper
            config to return a LlamaChatSession instance

        Parameters
        ----------
        conversation_id: str
        system_prompt: str, None
            system prompt for this session

        Returns
        -------
        LlamaChatSession
        """
        memory = self.memories.get(conversation_id)
        if not memory:
            memory = ChatMessageHistory(return_messages=True)
            self.memories[conversation_id] = memory

        return LlamaChatSession(self.llm, memory, system_prompt)
