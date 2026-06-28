from eaa_core.task_manager.commands import parse_user_input_command


def test_parse_user_input_command_recognizes_commands():
    assert parse_user_input_command("/exit").kind == "exit"
    assert parse_user_input_command("/return").kind == "return"
    assert parse_user_input_command("/chat").kind == "chat"
    assert parse_user_input_command("/skill").kind == "skill"
    skill = parse_user_input_command("/skill roi-search-workflow use it now")
    assert skill.kind == "skill"
    assert skill.argument == "roi-search-workflow"
    assert skill.text == "use it now"
    approval = parse_user_input_command("/setcodingtoolapproval false")
    assert approval.kind == "set_coding_tool_approval"
    assert approval.argument == "false"
    sandbox = parse_user_input_command("/setcodingtoolsandboxtype bubblewrap /tmp")
    assert sandbox.kind == "set_coding_tool_sandbox_type"
    assert sandbox.argument == "bubblewrap /tmp"


def test_parse_user_input_command_preserves_unknown_messages():
    message = parse_user_input_command("/unknown command")

    assert message.kind == "message"
    assert message.text == "/unknown command"
