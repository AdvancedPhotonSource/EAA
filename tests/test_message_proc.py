import pytest

from eaa_core.message_proc import complete_unresponded_tool_calls, get_message_preview


def test_get_message_preview_truncates_text():
    message = {"role": "assistant", "content": "abcdef"}

    assert get_message_preview(message, max_characters=3) == "abc"


def test_get_message_preview_uses_image_placeholder():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "abcdef"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
    }

    assert get_message_preview(message, max_characters=3) == "abc\n<image>"


def test_get_message_preview_rejects_negative_character_limit():
    with pytest.raises(ValueError, match="max_characters"):
        get_message_preview({"role": "user", "content": "text"}, max_characters=-1)


def test_complete_unresponded_tool_calls_uses_configurable_placeholder():
    context = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "tool_a", "arguments": "{}"},
                }
            ],
        }
    ]

    complete_unresponded_tool_calls(context, placeholder_content="call interrupted")

    assert context[-1] == {
        "role": "tool",
        "content": "call interrupted",
        "tool_call_id": "call_1",
    }
