from dataclasses import dataclass
from typing import Literal


UserCommandKind = Literal[
    "message",
    "exit",
    "return",
    "chat",
    "monitor",
    "help",
    "skill",
]


@dataclass
class UserInputCommand:
    """Parsed user input command.

    Parameters
    ----------
    kind : UserCommandKind
        Parsed command kind. Regular text is represented as ``"message"``.
    text : str, default=""
        Original user text for regular messages.
    argument : str, default=""
        Command argument text after the command token.
    """

    kind: UserCommandKind
    text: str = ""
    argument: str = ""


def parse_user_input_command(user_input: str) -> UserInputCommand:
    """Parse user input into a neutral command object.

    Parameters
    ----------
    user_input : str
        Raw user input.

    Returns
    -------
    UserInputCommand
        Parsed command. Unknown slash-prefixed text is treated as a regular
        message so it can still be sent to the model.
    """
    stripped = user_input.strip()
    command, _, remainder = stripped.partition(" ")
    command_lower = command.lower()
    argument = remainder.strip()
    if command_lower == "/exit" and argument == "":
        return UserInputCommand(kind="exit")
    if command_lower == "/return" and argument == "":
        return UserInputCommand(kind="return")
    if command_lower == "/chat" and argument == "":
        return UserInputCommand(kind="chat")
    if command_lower == "/help" and argument == "":
        return UserInputCommand(kind="help")
    if command_lower == "/skill" and argument == "":
        return UserInputCommand(kind="skill")
    if command_lower == "/monitor" and argument:
        return UserInputCommand(kind="monitor", argument=argument)
    return UserInputCommand(kind="message", text=user_input)
