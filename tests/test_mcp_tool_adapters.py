import json

import numpy as np

from eaa_core.tool.base import ExposedToolSpec
from eaa_core.tool.mcp_adapter import MCPParameterSettingProxy
from eaa_imaging.tool.imaging.acquisition import encode_image_array_payload
from eaa_imaging.tool.imaging.mcp_acquisition import MCPAcquireImageProxy


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


def test_mcp_acquire_image_proxy_updates_local_buffers_from_array_payload(tmp_path):
    calls = []
    remote_images = []

    def acquire_image(**kwargs):
        calls.append(kwargs)
        array = np.full((4, 5), kwargs["value"], dtype=np.float32)
        remote_images.append(array)
        return {
            "img_path": str(tmp_path / f"image_{len(calls)}.png"),
            "psize": kwargs["scan_step"],
        }

    def get_image_array_payload(buffer_name):
        if buffer_name in {"current", "image_k"}:
            image = remote_images[-1]
        elif buffer_name in {"previous", "image_km1"}:
            image = remote_images[-2]
        elif buffer_name in {"initial", "image_0"}:
            image = remote_images[0]
        else:
            raise ValueError(buffer_name)
        return encode_image_array_payload(image)

    def dump_array(buffer_name):
        raise AssertionError("Proxy dump_array should use the local buffered array.")

    remote = FakeRemoteTool(
        {
            "acquire_image": acquire_image,
            "get_image_array_payload": get_image_array_payload,
            "dump_array": dump_array,
        }
    )
    proxy = MCPAcquireImageProxy(remote)

    first = proxy.acquire_image(
        value=1,
        scan_step=0.5,
        y_center=1,
        x_center=2,
        size_y=4,
        size_x=5,
    )
    second = proxy.acquire_image(
        value=2,
        scan_step=0.25,
        y_center=3,
        x_center=4,
        size_y=4,
        size_x=5,
    )

    assert "array_path" not in first
    assert "array_path" not in second
    assert np.all(proxy.image_k == 2)
    assert np.all(proxy.image_km1 == 1)
    assert np.all(proxy.get_image_array("current") == 2)
    assert np.all(proxy.get_image_array("previous") == 1)
    assert proxy.psize_k == 0.25
    assert proxy.psize_km1 == 0.5
    assert "array_path" not in proxy.get_current_image_info()
    assert proxy.counter_acquire_image == 2
    assert proxy.image_acquisition_call_history[-1]["x_center"] == 4
    dump_spec = next(
        spec
        for spec in proxy.exposed_tools
        if spec.name == "dump_array"
    )
    dumped = dump_spec.function(buffer_name="previous")
    assert np.array_equal(
        np.load(dumped["array_path"], allow_pickle=False),
        proxy.get_image_array("previous"),
    )
    hidden_spec = next(
        spec
        for spec in proxy.exposed_tools
        if spec.name == "get_image_array_payload"
    )
    assert hidden_spec.model_visible is False


def test_mcp_parameter_setting_proxy_updates_local_history():
    received = []

    def set_parameters(parameters):
        received.append(parameters)
        return {"status": "ok", "parameters": parameters}

    remote = FakeRemoteTool({"set_parameters": set_parameters})
    proxy = MCPParameterSettingProxy(
        remote,
        parameter_names=["a", "b"],
        parameter_ranges=[(0, 0), (1, 1)],
    )

    result = proxy.set_parameters([0.2, 0.8])

    assert received == [[0.2, 0.8]]
    assert json.loads(result)["status"] == "ok"
    assert proxy.get_parameter_at_iteration(-1) == [0.2, 0.8]
