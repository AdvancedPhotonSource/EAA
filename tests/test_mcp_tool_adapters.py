import numpy as np
import pytest

from eaa_core.tool.base import BaseTool, ExposedToolSpec
from eaa_core.tool.mcp_adapter import (
    MCPRPCWrapper,
    set_parameters_with_local_history,
)


class FakeRemoteTool:
    def __init__(self, functions):
        self.functions = functions
        self.exposed_tools = [
            ExposedToolSpec(name=name, function=function)
            for name, function in functions.items()
        ]

    def __getattr__(self, name):
        if name in self.functions:
            return self.functions[name]
        raise AttributeError(name)


def test_base_tool_get_attribute_payload_returns_literals_and_arrays():
    class SampleTool(BaseTool):
        def build(self):
            self.scalar = 3
            self.array = np.arange(6, dtype=np.float32).reshape(2, 3)

    tool = SampleTool()
    payload_spec = next(
        spec
        for spec in tool.exposed_tools
        if spec.name == "sample_tool.get_attribute_payload"
    )

    assert payload_spec.model_visible is False
    assert tool.get_attribute_payload("scalar") == 3
    assert np.array_equal(
        BaseTool.decode_array_payload(tool.get_attribute_payload("array")),
        tool.array,
    )


def test_mcp_rpc_wrapper_maps_arguments_and_syncs_attributes():
    calls = []
    remote_images = []
    remote_history = []
    remote_psize = None

    def acquire_2d_image(**kwargs):
        nonlocal remote_psize
        calls.append(kwargs)
        array = np.full((4, 5), kwargs["value"], dtype=np.float32)
        remote_images.append(array)
        remote_psize = kwargs["scan_step"]
        remote_history.append(
            {
                "x_center": kwargs["x_center"],
                "y_center": kwargs["y_center"],
                "size_x": kwargs["width"],
                "size_y": kwargs["height"],
            }
        )
        return {
            "img_path": f"image_{len(calls)}.png",
            "psize": kwargs["scan_step"],
        }

    def get_attribute_payload(name):
        if name == "image_k":
            return BaseTool.encode_array_payload(remote_images[-1])
        if name == "psize_k":
            return remote_psize
        if name == "image_acquisition_call_history":
            return remote_history
        raise ValueError(name)

    remote = FakeRemoteTool(
        {
            "acquisition_server.acquire_2d_image": acquire_2d_image,
            "acquisition_server.get_attribute_payload": get_attribute_payload,
        }
    )
    wrapper = MCPRPCWrapper(
        mcp_tool_client=remote,
        mappings={
            "acquire_image": {
                "remote": "acquire_2d_image",
                "arguments": {"size_x": "width", "size_y": "height"},
                "sync": {
                    "image_k": "image_k",
                    "psize_k": "psize_k",
                    "image_acquisition_call_history": "image_acquisition_call_history",
                },
            },
        },
    )

    first = wrapper.acquire_image(
        value=1,
        scan_step=0.5,
        y_center=1,
        x_center=2,
        size_y=4,
        size_x=5,
    )
    second = wrapper.acquire_image(
        value=2,
        scan_step=0.25,
        y_center=3,
        x_center=4,
        size_y=4,
        size_x=5,
    )

    assert first["img_path"] == "image_1.png"
    assert second["img_path"] == "image_2.png"
    assert calls[0]["width"] == 5
    assert calls[0]["height"] == 4
    assert "size_x" not in calls[0]
    assert np.all(wrapper.image_k == 2)
    assert wrapper.psize_k == 0.25
    assert wrapper.image_acquisition_call_history[-1]["x_center"] == 4


def test_mcp_rpc_wrapper_raises_on_ambiguous_suffix_match():
    remote = FakeRemoteTool(
        {
            "server_a.acquire_image": lambda **kwargs: kwargs,
            "server_b.acquire_image": lambda **kwargs: kwargs,
        }
    )
    wrapper = MCPRPCWrapper(
        mcp_tool_client=remote,
        mappings={"acquire_image": {"remote": "acquire_image"}},
    )

    with pytest.raises(AttributeError, match="ambiguous"):
        wrapper.acquire_image()


def test_mcp_rpc_wrapper_raises_when_sync_support_tool_is_missing():
    remote = FakeRemoteTool({"acquire_image": lambda **kwargs: {}})
    wrapper = MCPRPCWrapper(
        mcp_tool_client=remote,
        mappings={
            "acquire_image": {
                "remote": "acquire_image",
                "sync": {"image_k": "image_k"},
            },
        },
    )

    with pytest.raises(AttributeError, match="get_attribute_payload"):
        wrapper.acquire_image()


def test_parameter_setting_rpc_wrapper_updates_local_history():
    received = []

    def set_parameters(parameters):
        received.append(parameters)
        return {"status": "ok", "parameters": parameters}

    remote = FakeRemoteTool({"set_parameters": set_parameters})
    wrapper = MCPRPCWrapper(
        mcp_tool_client=remote,
        mappings={
            "set_parameters": {
                "remote": "set_parameters",
                "arguments": {"parameters": "parameters"},
            },
        },
        initial_attributes={
            "parameter_names": ["a", "b"],
            "parameter_ranges": [(0, 0), (1, 1)],
            "parameter_history": {"a": [], "b": []},
        },
        local_methods={"set_parameters": set_parameters_with_local_history},
    )

    result = wrapper.set_parameters(np.array([0.2, 0.8]))

    assert received == [[0.2, 0.8]]
    assert result["status"] == "ok"
    assert wrapper.len_parameter_history == 1
    assert wrapper.get_parameter_at_iteration(-1) == [0.2, 0.8]
