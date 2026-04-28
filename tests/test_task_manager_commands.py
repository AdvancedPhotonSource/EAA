from eaa_core.task_manager.commands import parse_user_input_command


def test_parse_user_input_command_recognizes_commands():
    assert parse_user_input_command("/exit").kind == "exit"
    assert parse_user_input_command("/return").kind == "return"
    assert parse_user_input_command("/chat").kind == "chat"
    assert parse_user_input_command("/help").kind == "help"
    assert parse_user_input_command("/skill").kind == "skill"


def test_parse_user_input_command_preserves_messages_and_monitor_argument():
    message = parse_user_input_command("/unknown command")
    monitor = parse_user_input_command("/monitor check beam drift")

    assert message.kind == "message"
    assert message.text == "/unknown command"
    assert monitor.kind == "monitor"
    assert monitor.argument == "check beam drift"
