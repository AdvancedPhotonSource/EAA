from dataclasses import dataclass
from typing import Literal


UserCommandKind = Literal[
    "message",
    "exit",
    "return",
    "chat",
    "skill",
    "set_coding_tool_approval",
    "set_coding_tool_sandbox_type",
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
    if command_lower == "/skill":
        skill_name, _, remaining_text = argument.partition(" ")
        return UserInputCommand(
            kind="skill",
            argument=skill_name.strip(),
            text=remaining_text.strip(),
        )
    if command_lower == "/setcodingtoolapproval":
        return UserInputCommand(kind="set_coding_tool_approval", argument=argument)
    if command_lower == "/setcodingtoolsandboxtype":
        return UserInputCommand(kind="set_coding_tool_sandbox_type", argument=argument)
    return UserInputCommand(kind="message", text=user_input)
