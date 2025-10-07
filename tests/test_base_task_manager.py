import atexit
import random
import string
import unittest.mock as mock
from unittest.mock import MagicMock, call
from typing import Callable, Dict, Any, Iterable, Iterator, Tuple

import pytest

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import ToolReturnType
from eaa.api.llm_config import OpenAIConfig

import test_utils as tutils


def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_message(role: str = "assistant", content: str = None) -> Dict[str, Any]:
    """Generate a random message dictionary."""
    message = {
        "role": role,
        "content": content or generate_random_string(50),
    }
    # Add tool_calls field only for assistant messages
    if role == "assistant":
        message["tool_calls"] = []
    return message


def generate_random_tool_response(content: str = None, tool_call_id: str = None) -> Dict[str, Any]:
    """Generate a random tool response."""
    return {
        "role": "tool",
        "content": content or generate_random_string(30),
        "tool_call_id": tool_call_id or generate_random_string(10)
    }


def make_receive_side_effect(
    responses: Iterable[Tuple[Dict[str, Any], Dict[str, Any] | None]]
) -> Callable:
    """Create a side effect that adapts to return_outgoing_message."""

    response_iter: Iterator[Tuple[Dict[str, Any], Dict[str, Any] | None]] = iter(responses)

    def _side_effect(*args, **kwargs):
        try:
            response, outgoing = next(response_iter)
        except StopIteration as exc:
            raise StopIteration("agent.receive was called more times than expected") from exc

        if kwargs.get("return_outgoing_message", True):
            return response, outgoing
        return response

    return _side_effect


def _fake_generate_openai_message(
    content,
    role="user",
    tool_call_id=None,
    image=None,
    image_path=None,
    encoded_image=None,
):
    if sum(
        [
            image is not None,
            image_path is not None,
            encoded_image is not None,
        ]
    ) > 1:
        raise ValueError(
            "Only one of `image`, `encoded_image`, or `image_path` should be provided."
        )

    message = {"role": role}
    if role == "tool":
        message["tool_call_id"] = tool_call_id

    if image is not None or image_path is not None or encoded_image is not None:
        encoded_value = encoded_image or "FAKE_BASE64_IMAGE"
        message["content"] = [
            {"type": "text", "text": content},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_value}"
                },
            },
        ]
    else:
        message["content"] = content

    return message


_generate_openai_patch = mock.patch(
    "eaa.task_managers.base.generate_openai_message",
    _fake_generate_openai_message,
)
_generate_openai_patch.start()
atexit.register(_generate_openai_patch.stop)


class TestBaseTaskManagerFeedbackLoop(tutils.BaseTester):

    def setup_method(
        self,
        name="",
        generate_data=False,
        generate_gold=False,
        debug=False,
        action=None,
        pytestconfig=None,
    ):
        """Set up test fixtures."""
        super().setup_method(
            name=name,
            generate_data=generate_data,
            generate_gold=generate_gold,
            debug=debug,
            action=action,
            pytestconfig=pytestconfig,
        )
        
        # Create a mock LLM config
        self.mock_llm_config = OpenAIConfig(
            model="gpt-4",
            api_key="fake-key",
            base_url="https://api.openai.com/v1"
        )
        
        # Create task manager with mocked dependencies
        self.task_manager = BaseTaskManager(
            llm_config=self.mock_llm_config,
            tools=[],
            build=False  # Don't build to avoid real dependencies
        )
        
        # Mock the agent
        self.task_manager.agent = MagicMock()

        # Mock message history methods
        self.task_manager.update_message_history = MagicMock()
        self.task_manager.get_user_input = MagicMock()

    def test_run_feedback_loop_with_image_path_tool_response(self):
        """Test feedback loop with normal IMAGE_PATH tool response."""
        # Setup
        initial_prompt = generate_random_string(30)
        initial_image_path = "/path/to/initial/image.jpg"
        max_rounds = 2
        
        # Mock agent responses
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content="TERMINATE - Task completed")
        outgoing2 = generate_random_message(role="user")
        
        response3 = generate_random_message(content="Continuing after user input")
        outgoing3 = generate_random_message(role="user")
        
        # Mock tool responses
        tool_response = generate_random_tool_response(content="/path/to/generated/image.jpg")
        tool_responses = [tool_response]
        tool_response_types = [ToolReturnType.IMAGE_PATH]
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (response2, None),  # After tool response - contains TERMINATE
            (response3, outgoing3),  # After user continues with "\exit"
        ])
        
        # Configure tool call handler to return tool response first, then empty for termination
        self.task_manager.agent.handle_tool_call.side_effect = [
            (tool_responses, tool_response_types),  # First tool call
            ([], []),  # No tool calls in termination response
        ]
        
        self.task_manager.get_user_input.return_value = "\\exit"
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=initial_image_path,
            max_rounds=max_rounds
        )

        # Verify agent.receive calls
        expected_receive_calls = [
            call(initial_prompt, context=[], image_path=initial_image_path, return_outgoing_message=True),
            call(message=None, image_path=None, context=[], return_outgoing_message=False),
        ]
        self.task_manager.agent.receive.assert_has_calls(expected_receive_calls)

        # Verify the generated image follow-up message was added to history
        expected_image_followup = _fake_generate_openai_message(
            content="Here is the image the tool returned.",
            image_path="/path/to/generated/image.jpg",
        )
        self.task_manager.update_message_history.assert_any_call(
            expected_image_followup,
            update_context=True,
            update_full_history=True,
        )

        # Verify handle_tool_call was called appropriately
        assert self.task_manager.agent.handle_tool_call.call_count == 1  # Only first response checked for tool calls

        # Verify update_message_history was called appropriately
        assert self.task_manager.update_message_history.call_count >= 4  # Initial + response + tool response + image response
        
        # Verify user was prompted when TERMINATE condition triggered
        self.task_manager.get_user_input.assert_called_once()

    def test_run_feedback_loop_with_exception_tool_response(self):
        """Test feedback loop with EXCEPTION tool response."""
        # Setup
        initial_prompt = generate_random_string(30)
        
        # Mock agent responses
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content="TERMINATE")
        outgoing2 = generate_random_message(role="user")
        
        # Mock tool responses with exception
        tool_response = generate_random_tool_response(content="Error: " + generate_random_string(20))
        tool_responses = [tool_response]
        tool_response_types = [ToolReturnType.EXCEPTION]
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (response2, None),  # After exception handling
        ])
        
        self.task_manager.agent.handle_tool_call.return_value = (tool_responses, tool_response_types)
        self.task_manager.get_user_input.return_value = "\\exit"
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=3
        )

        # Verify the tool response was recorded
        self.task_manager.update_message_history.assert_any_call(
            tool_response,
            update_context=True,
            update_full_history=True,
        )

        # Verify the follow-up receive call used context replay
        self.task_manager.agent.receive.assert_has_calls([
            call(initial_prompt, context=[], image_path=None, return_outgoing_message=True),
            call(message=None, image_path=None, context=[], return_outgoing_message=False),
        ])
        
    def test_run_feedback_loop_with_multiple_tool_calls(self):
        """Test feedback loop with multiple tool calls (error condition)."""
        # Setup
        initial_prompt = generate_random_string(30)
        
        # Mock agent responses
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content="TERMINATE")
        outgoing2 = generate_random_message(role="user")
        
        # Mock multiple tool responses
        tool_response1 = generate_random_tool_response()
        tool_response2 = generate_random_tool_response()
        tool_responses = [tool_response1, tool_response2]
        tool_response_types = [ToolReturnType.TEXT, ToolReturnType.TEXT]
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (response2, outgoing2),  # After error handling
        ])
        
        self.task_manager.agent.handle_tool_call.return_value = (tool_responses, tool_response_types)
        self.task_manager.get_user_input.return_value = "\\exit"
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=3
        )
        
        # Verify multiple tool calls error message was sent
        self.task_manager.agent.receive.assert_any_call(
            "There are more than one tool calls in your response. "
            "Make sure you only make one call at a time. Please redo "
            "your tool calls.",
            image_path=None,
            context=[],
            return_outgoing_message=True
        )
        
    def test_run_feedback_loop_with_no_tool_calls(self):
        """Test feedback loop with no tool calls (error condition)."""
        # Setup
        initial_prompt = generate_random_string(30)
        
        # Mock agent responses
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content="TERMINATE")
        outgoing2 = generate_random_message(role="user")
        
        # Mock no tool responses
        tool_responses = []
        tool_response_types = []
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (response2, outgoing2),  # After error handling
        ])
        
        self.task_manager.agent.handle_tool_call.return_value = (tool_responses, tool_response_types)
        self.task_manager.get_user_input.return_value = "\\exit"
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=3
        )
        
        # Verify no tool calls error message was sent
        self.task_manager.agent.receive.assert_any_call(
            "There is no tool call in the response. Make sure you call the tool correctly. "
            "If you need human intervention, say \"TERMINATE\".",
            image_path=None,
            context=[],
            return_outgoing_message=True
        )
        
    def test_run_feedback_loop_with_non_image_tool_responses_disallowed(self):
        """Test feedback loop with non-image tool responses when disallowed."""
        # Setup
        initial_prompt = generate_random_string(30)
        
        # Mock agent responses
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content="TERMINATE")
        outgoing2 = generate_random_message(role="user")
        
        # Mock non-image tool response
        tool_response = generate_random_tool_response(content=generate_random_string(20))
        tool_responses = [tool_response]
        tool_response_types = [ToolReturnType.TEXT]
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (response2, None),  # After error handling
        ])
        
        self.task_manager.agent.handle_tool_call.return_value = (tool_responses, tool_response_types)
        self.task_manager.get_user_input.return_value = "\\exit"
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=3,
            allow_non_image_tool_responses=False
        )
        
        # Verify non-image tool response error message was sent
        expected_message = _fake_generate_openai_message(
            content=
            f"The tool should return an image path, but got {str(ToolReturnType.TEXT)}. "
            "Make sure you call the right tool correctly.",
        )
        self.task_manager.update_message_history.assert_any_call(
            expected_message,
            update_context=True,
            update_full_history=True,
        )

        self.task_manager.agent.receive.assert_has_calls([
            call(initial_prompt, context=[], image_path=None, return_outgoing_message=True),
            call(message=None, image_path=None, context=[], return_outgoing_message=False),
        ])
        
    def test_run_feedback_loop_with_non_image_tool_responses_allowed(self):
        """Test feedback loop with non-image tool responses when allowed."""
        # Setup
        initial_prompt = generate_random_string(30)
        
        # Mock agent responses
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content="TERMINATE")
        outgoing2 = generate_random_message(role="user")
        
        # Mock non-image tool response
        tool_response = generate_random_tool_response(content=generate_random_string(20))
        tool_responses = [tool_response]
        tool_response_types = [ToolReturnType.TEXT]
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (response2, None),  # After processing tool response
        ])
        
        self.task_manager.agent.handle_tool_call.return_value = (tool_responses, tool_response_types)
        self.task_manager.get_user_input.return_value = "\\exit"
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=3,
            allow_non_image_tool_responses=True
        )
        
        # Verify tool response was processed normally (no error message) and context replay occurred
        self.task_manager.agent.receive.assert_has_calls([
            call(initial_prompt, context=[], image_path=None, return_outgoing_message=True),
            call(message=None, image_path=None, context=[], return_outgoing_message=False),
        ])
        
    def test_run_feedback_loop_max_rounds_reached(self):
        """Test feedback loop reaching maximum rounds."""
        # Setup
        initial_prompt = generate_random_string(30)
        max_rounds = 2
        
        # Mock agent responses - never terminate
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content=generate_random_string(20))
        outgoing2 = generate_random_message(role="user")
        
        response3 = generate_random_message(content=generate_random_string(20))
        outgoing3 = generate_random_message(role="user")
        
        # Mock tool responses
        tool_response = generate_random_tool_response(content="/path/to/image.jpg")
        tool_responses = [tool_response]
        tool_response_types = [ToolReturnType.IMAGE_PATH]
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (response2, None),  # Round 1
            (response3, None),  # Round 2
        ])
        
        self.task_manager.agent.handle_tool_call.side_effect = [
            (tool_responses, tool_response_types),  # Round 1
            (tool_responses, tool_response_types),  # Round 2
        ]
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=max_rounds
        )

        # Verify that the loop stops after max_rounds
        # Should have exactly 3 receive calls (initial + 2 rounds)
        assert self.task_manager.agent.receive.call_count == 3
        assert self.task_manager.agent.receive.call_args_list[1] == call(
            message=None,
            image_path=None,
            context=[],
            return_outgoing_message=False,
        )
        assert self.task_manager.agent.receive.call_args_list[2] == call(
            message=None,
            image_path=None,
            context=[],
            return_outgoing_message=False,
        )
        
    def test_run_feedback_loop_with_hook_function(self):
        """Test feedback loop with custom hook function."""
        # Setup
        initial_prompt = generate_random_string(30)
        
        # Mock agent responses
        response1 = generate_random_message(content=generate_random_string(20))
        outgoing1 = generate_random_message(role="user")
        
        # Mock hook function
        hook_response = generate_random_message(content="Hook response")
        hook_outgoing = generate_random_message(role="user")
        mock_hook = MagicMock(return_value=(hook_response, hook_outgoing))
        
        # Mock tool responses
        tool_response = generate_random_tool_response(content="/path/to/image.jpg")
        tool_responses = [tool_response]
        tool_response_types = [ToolReturnType.IMAGE_PATH]
        
        # Configure mocks
        followup_response = generate_random_message(content=generate_random_string(20))

        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call
            (followup_response, None),  # Follow-up after hook processing
        ])
        
        self.task_manager.agent.handle_tool_call.return_value = (tool_responses, tool_response_types)
        
        # Execute with hook function
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=1,
            hook_functions={"image_path_tool_response": mock_hook}
        )
        
        # Verify hook function was called with correct image path
        mock_hook.assert_called_once_with("/path/to/image.jpg")
        
        # Verify agent.receive was called once for the initial turn and once for context replay
        assert self.task_manager.agent.receive.call_count == 2
        assert self.task_manager.agent.receive.call_args_list[1] == call(
            message=None,
            image_path=None,
            context=[],
            return_outgoing_message=False,
        )

    def test_run_feedback_loop_terminate_user_continues(self):
        """Test feedback loop when user chooses to continue after TERMINATE."""
        # Setup
        initial_prompt = generate_random_string(30)
        user_continuation = generate_random_string(25)
        
        # Mock agent responses
        response1 = generate_random_message(content="TERMINATE - Done")
        outgoing1 = generate_random_message(role="user")
        
        response2 = generate_random_message(content="Continuing...")
        outgoing2 = generate_random_message(role="user")
        
        response3 = generate_random_message(content="TERMINATE")
        outgoing3 = generate_random_message(role="user")
        
        # Configure mocks
        self.task_manager.agent.receive.side_effect = make_receive_side_effect([
            (response1, outgoing1),  # Initial call with TERMINATE
            (response2, outgoing2),  # User continuation
            (response3, outgoing3),  # Final TERMINATE
        ])
        
        self.task_manager.agent.handle_tool_call.side_effect = [
            ([], []),  # No tool calls after continuation
        ]
        
        self.task_manager.get_user_input.side_effect = [user_continuation, "\\exit"]
        
        # Execute
        self.task_manager.run_feedback_loop(
            initial_prompt=initial_prompt,
            max_rounds=5
        )
        
        # Verify user was prompted for continuation
        assert self.task_manager.get_user_input.call_count == 2
        
        # Verify agent received the user continuation
        self.task_manager.agent.receive.assert_any_call(
            user_continuation,
            context=[],
            image_path=None,
            return_outgoing_message=True
        )


if __name__ == "__main__":
    tester = TestBaseTaskManagerFeedbackLoop()
    test_methods = [
        "test_run_feedback_loop_with_image_path_tool_response",
        "test_run_feedback_loop_with_exception_tool_response",
        "test_run_feedback_loop_with_multiple_tool_calls",
        "test_run_feedback_loop_with_no_tool_calls",
        "test_run_feedback_loop_with_non_image_tool_responses_disallowed",
        "test_run_feedback_loop_with_non_image_tool_responses_allowed",
        "test_run_feedback_loop_max_rounds_reached",
        "test_run_feedback_loop_with_hook_function",
        "test_run_feedback_loop_terminate_user_continues",
    ]
    for method in test_methods:
        tester.setup_method()
        getattr(tester, method)()
