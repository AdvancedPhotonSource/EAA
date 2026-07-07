"""FastMCP HTTP server that forwards simulated acquisition tools to ZMQ."""

from __future__ import annotations

from typing import Annotated, Any
import argparse
import asyncio
import logging

from fastmcp import FastMCP

from control_suite_mcp_dummy.config import load_config, resolve_setting
from control_suite_mcp_dummy.zmq_client import WorkerClient

logger = logging.getLogger(__name__)


def create_mcp(worker_endpoint: str, timeout_ms: int = 30_000) -> FastMCP:
    """Create the FastMCP server.

    Parameters
    ----------
    worker_endpoint
        ZMQ endpoint used to reach the instrument worker.
    timeout_ms
        Worker request timeout in milliseconds.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    mcp = FastMCP("Control Suite MCP Dummy")
    client = WorkerClient(worker_endpoint, timeout_ms=timeout_ms)

    async def call_worker(method: str, params: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(client.call, method, params)

    @mcp.tool()
    async def health() -> dict[str, str]:
        """Check whether the instrument worker is reachable."""
        return await call_worker("health")

    @mcp.tool()
    async def get_state() -> dict[str, Any]:
        """Return JSON-serializable simulated instrument state and artifact metadata."""
        return await call_worker("get_state")

    @mcp.tool()
    async def set_blur(
        blur: Annotated[float | None, "Gaussian blur sigma applied to sampled data."],
    ) -> dict[str, Any]:
        """Set the simulated image blur sigma in image-pixel units."""
        return await call_worker("set_blur", {"blur": blur})

    @mcp.tool()
    async def set_offset(
        y_offset: Annotated[float, "Simulated y drift offset."],
        x_offset: Annotated[float, "Simulated x drift offset."],
    ) -> dict[str, Any]:
        """Set simulated drift offset in y, x order."""
        return await call_worker("set_offset", {"y_offset": y_offset, "x_offset": x_offset})

    @mcp.tool()
    async def set_config(
        name: Annotated[str, "Writable simulated instrument configuration name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Set a writable simulated instrument configuration value.

        This changes simulation settings such as noise, plotting, line-scan fit
        options, parameter names, and tuning ranges. Values must be
        JSON-serializable and are applied inside the worker process.
        """
        return await call_worker("set_config", {"name": name, "value": value})

    @mcp.tool()
    async def set_attribute(
        name: Annotated[str, "Writable simulated instrument configuration name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Alias for ``set_config`` used by the EAA MCP acquisition proxy."""
        return await call_worker("set_attribute", {"name": name, "value": value})

    @mcp.tool()
    async def set_parameters(
        parameters: Annotated[
            list[float],
            "Simulated tuning parameter values to set.",
        ],
    ) -> str:
        """Set simulated instrument tuning parameters.

        The parameter order must match the worker's configured
        ``parameter_names``. Values outside configured ranges are rejected by
        the worker before any simulated state is updated.
        """
        return await call_worker("set_parameters", {"parameters": parameters})

    @mcp.tool()
    async def acquire_image(
        x_center: Annotated[float, "The x-coordinate of the center of the image to acquire."],
        y_center: Annotated[float, "The y-coordinate of the center of the image to acquire."],
        size_y: Annotated[int, "The height of the image to acquire."],
        size_x: Annotated[int, "The width of the image to acquire."],
        scan_step: Annotated[
            float,
            "The step size between sampled points in both y and x directions.",
        ] = 1.0,
    ) -> dict[str, Any]:
        """Acquire an image from the simulated instrument.

        Coordinates are supplied in ``x_center``, ``y_center`` order. The
        worker returns ``img_path`` for the PNG display artifact and ``psize``
        for pixel size. Use ``dump_array`` to export numerical image data from
        worker buffers.
        """
        return await call_worker(
            "acquire_image",
            {
                "x_center": x_center,
                "y_center": y_center,
                "size_y": size_y,
                "size_x": size_x,
                "scan_step": scan_step,
            },
        )

    @mcp.tool()
    async def acquire_line_scan(
        x_center: Annotated[float, "The x-coordinate of the center of the line scan."],
        y_center: Annotated[float, "The y-coordinate of the center of the line scan."],
        length: Annotated[float, "The length of the line scan."],
        scan_step: Annotated[float, "The step size of the line scan."],
        angle: Annotated[
            float,
            "The angle of the line scan in degrees. 0 is horizontal.",
        ] = 0.0,
    ) -> dict[str, Any]:
        """Acquire a line scan from the simulated instrument.

        Coordinates are supplied in ``x_center``, ``y_center`` order. Angle is
        in degrees, where 0 is horizontal and positive angles rotate
        counter-clockwise. The result includes ``img_path`` and numeric
        ``fwhm``; optional Gaussian-fit metadata is controlled by worker config.
        """
        return await call_worker(
            "acquire_line_scan",
            {
                "x_center": x_center,
                "y_center": y_center,
                "length": length,
                "scan_step": scan_step,
                "angle": angle,
            },
        )

    @mcp.tool()
    async def dump_array(
        buffer_name: Annotated[
            str,
            "Image buffer name to dump. Must be image_k, image_km1, or image_0.",
        ],
    ) -> dict[str, Any]:
        """Save a server-side image buffer as a ``.npy`` artifact.

        This returns metadata with ``path`` and avoids embedding large arrays
        in the model-visible response. Artifact paths must be readable by the
        MCP client process.
        """
        return await call_worker("dump_array", {"buffer_name": buffer_name})

    @mcp.tool()
    async def get_attribute_payload(
        name: Annotated[str, "Native worker attribute name to fetch."],
    ) -> Any:
        """Return a JSON payload for a worker attribute.

        Literal values are returned directly. NumPy arrays are encoded in an
        ``encoded_data`` object containing ``type``, ``dtype``, ``shape``, and
        ``data`` fields for logic-driven EAA adapters.
        """
        return await call_worker("get_attribute_payload", {"name": name})

    return mcp


def build_parser() -> argparse.ArgumentParser:
    """Build the MCP server CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML MCP configuration.")
    parser.add_argument("--worker", default=None)
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--path", default=None)
    return parser


def main() -> None:
    """Run the FastMCP HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    config = load_config(args.config)
    worker = resolve_setting(config, "worker", "endpoint", args.worker, "tcp://127.0.0.1:5555")
    timeout_ms = resolve_setting(config, "worker", "request_timeout_ms", args.timeout_ms, 30_000)
    host = resolve_setting(config, "mcp", "host", args.host, "127.0.0.1")
    port = resolve_setting(config, "mcp", "port", args.port, 8050)
    path = resolve_setting(config, "mcp", "path", args.path, "/mcp")
    logger.info("Starting MCP server, forwarding to worker at %s", worker)
    mcp = create_mcp(worker, timeout_ms=timeout_ms)
    mcp.run(transport="http", host=host, port=port, path=path)


if __name__ == "__main__":
    main()
