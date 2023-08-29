
from ChatPromptWrapper import MessageRole

def print_chat_colored(messages):
    """ Prints the chat colored

    Parameters
    ----------
    messages: List[Message]
        list of Message instancesk
    """
    reset = "\033[00m"
    color_map = {
        MessageRole.SYSTEM: ("\033[1;35m", "\033[35m"),
        MessageRole.USER: ("\033[1;33m", "\033[33m"),
        MessageRole.ASSISTANT: ("\033[1;31m", "\033[31m"),
    }
    for msg in messages:
        role_color, content_color = color_map[msg["role"]]
        formatted_message = f"{role_color}{msg['role'].name : >12}{reset}> {content_color}{msg['msg']}{reset}"

        print(formatted_message)
