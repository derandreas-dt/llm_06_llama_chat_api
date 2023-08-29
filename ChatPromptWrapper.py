from enum import Enum
from collections import UserDict

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

class MessageRole(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3

    @classmethod
    def has_value(self, val):
        return val in self._value2member_map_

    @classmethod
    def has_key(self, key):
        return key in self._member_names_

    @classmethod
    def as_str(self):
        print(self.value)
        print(self.name)
        return self._member_names_[val]


class Message(UserDict):
    """ definition of a single message.

        A message can be a system message (system prompt)
        or a user message (question) or a assistant message
        (ai response).
    """
    def __init__(self, role, msg):
        """ Init a new Message by defining its role and content

        Parameters
        ----------
        role: MessageRole
        msg: str
        """
        super().__init__({
            "role": role,
            "msg": msg
        })

class LlamaChatSession:
    """ Session to interact with the llm and replay
        the history of the conversation in each new
        query to the llm. Handles the initial build
        of a system prompt and the format of new
        messages.
    """
    def __init__(self, llm, system_prompt):
        """ Init the session, setup everything based
            on the input of this constructor

        Parameters
        ----------
        llm: LlamaCpp
            instance of the llama cpp model
        system_prompt: str
            system prompt for this session
        """
        self.llm = llm
        self.messages = []

        self.messages = [
            {
                "role": MessageRole.SYSTEM,
                "msg": system_prompt or DEFAULT_SYSTEM_PROMPT,
            }
        ]

    def __call__(self, message, max_tokens = 128):
        """ Query the AI with a new message

        This will prepare the message in the way that the AI
        understands its best with the history of the chat.
        Each response is added to the chat history and reused
        later in the next queries.

        The interaction with the LLM is done with the `generate`
        call, instead of directly calling the instance.
        The `generate` expects not strings, rather than the encoded
        tokens as int form the vocab dict. The input is List[Int].

        Parameters
        ----------
        message: str
        max_tokens: int
            maximal tokens in the response

        Returns
        -------
        str
        """
        # append the new message to the history
        self.messages.append(Message(MessageRole.USER, message))

        # convert the chat history into tokens, which is then fed
        # to the AI to calculate the response
        message_tokens = self.prepare_messages(self.messages)
        response = self.llm.client.generate(message_tokens)

        # check if we hit the max tokens limit
        max_tokens = (
            max_tokens if max_tokens + len(message_tokens) < self.llm.n_ctx else (self.llm.n_ctx - len(messages_tokens))
        )
        result = []
        # iterate over the response and detokenize the resulting
        # tokens (int) into words again,w hich then can be used later
        for i, token in enumerate(response):
            if max_tokens == i or token == self.llm.client.token_eos():
                break
            result.append(self.llm.client.detokenize([token]).decode("utf-8"))

        result = "".join(result).strip()

        # append the AI response to the chat history
        self.messages.append(Message(MessageRole.ASSISTANT, result))

        # return the response
        return result

    def get_messages(self, start=None, stop=None, step=None):
        """ Returns the messages

            if the start/stop/step params are given, it will
            handle them like the python list index accessing
            using slice() function internally.

        Parameters
        ----------
        start: int
        stop: int
        step: int

        Returns
        -------
        list
        """
        if any([start, stop, step]):
            return self.messages[slice(start, stop, step)]

        return self.messages

    def prepare_messages(self, messages):
        """ Prepare the message that will be put into the AI

        This builds the complete history based on a system message,
        the user questions and AIs answers as formated string.
        The string is encoded into tokens (List[Int]) to be used
        later to call `generate` on the LLM.

        This formats with <<SYS>> and [INST] as well as BOS/EOS tags.

        Parameters
        ----------
        messages: List[Message]
            list of Messages of the chat history

        Returns
        -------
        List[Int]
            tokenized chat history
        """
        # first put the system prompt
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "msg": B_SYS + messages[0]["msg"] + E_SYS + messages[1]["msg"],
            }
        ] + messages[2:]

        # all messages here should be in order of user - ai - user - ai - user ...
        message_tokens = sum(
        [
            self.tokenizer_encode(
                f"{B_INST} {prompt['msg'].strip()} {E_INST} {answer['msg'].strip()} ",
                True,
                True
            )
            for prompt, answer in zip(messages[::2], messages[1::2])
        ],
        [],
        )

        message_tokens += self.tokenizer_encode(
            f"{B_INST} {messages[-1]['msg'].strip()} {E_INST}",
            True,
            False,
        )

        return message_tokens

    def tokenizer_encode(self, msg, bos=False, eos=False):
        """ Encode the tokens to token ids for later use
            in the `generate` function.

            Then it will handle the BOS/EOS indicator.

        Parameters
        ----------
        msg: str
            the message to tokenize
        bos: boolean, default False
        eos: boolean, default False

        Returns
        -------
        List[Int]
        """
        tokens = self.llm.client.tokenize(text=b" " + bytes(msg, encoding="utf-8"), add_bos=False)

        if bos:
            tokens = [self.llm.client.token_bos()] + tokens

        if eos:
            tokens = tokens + [self.llm.client.token_eos()]

        return tokens

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

    def new_session(self, system_prompt=None):
        """ Creates a new Session by using the wrapper
            config to return a LlamaChatSession instance

        Parameters
        ----------
        system_prompt: str, None
            system prompt for this session

        Returns
        -------
        LlamaChatSession
        """
        return LlamaChatSession(self.llm, system_prompt)

