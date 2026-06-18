from PIL import Image
import pytest

from eaa_core.message_proc import (
    complete_unresponded_tool_calls,
    convert_tagged_text_to_openai_message,
    get_image_paths_from_text,
    get_message_preview,
)


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


def test_get_image_paths_from_text_extracts_paths_and_cleaned_text():
    paths, cleaned_text = get_image_paths_from_text(
        "before <img /tmp/a.png> middle <img /tmp/b.png> after",
        return_text_without_image_tag=True,
    )

    assert paths == ["/tmp/a.png", "/tmp/b.png"]
    assert cleaned_text == "before  middle  after"


def test_convert_tagged_text_to_openai_message_keeps_plain_text_plain():
    message = convert_tagged_text_to_openai_message("hello")

    assert message == {"role": "user", "content": "hello"}


def test_convert_tagged_text_to_openai_message_builds_multimodal_message(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(image_path)

    message = convert_tagged_text_to_openai_message(f"look <img {image_path}>")

    assert message["role"] == "user"
    assert message["content"][0] == {"type": "text", "text": "look "}
    assert message["content"][1]["type"] == "image_url"
    assert message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
