from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

def print_memory_colored(memory):
    """ Prints the ChatMessageHistory in
        pretty format with colors

    Parameters
    ----------
    memory: ChatMessageHistory

    """
    reset = "\033[00m"
    color_map = {
        'system': ("\033[1;35m", "\033[35m"),
        'human': ("\033[1;33m", "\033[33m"),
        'assistant': ("\033[1;31m", "\033[31m"),
    }
    for message in memory.messages:
        if isinstance(message, HumanMessage):
            role_color, content_color = color_map['human']
            formatted_message = f"{role_color}Human{reset}> {content_color}{message.content}{reset}"
        elif isinstance(message, AIMessage):
            role_color, content_color = color_map['assistant']
            formatted_message = f"{role_color}Assistant{reset}> {content_color}{message.content}{reset}"
        elif isinstance(message, SystemMessage):
            role_color, content_color = color_map['system']
            formatted_message = f"{role_color}System{reset}> {content_color}{message.content}{reset}"
        else:
            raise ValueError(f"Got unknown type {message}")

        print(formatted_message)
