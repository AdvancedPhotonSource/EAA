"""ZMQ instrument worker for the dummy control suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import argparse
import json
import logging

import numpy as np
import tifffile
import zmq

from control_suite_mcp_dummy.config import get_config_section, load_config, resolve_setting
from control_suite_mcp_dummy.protocol import error_response, ok_response
from control_suite_mcp_dummy.simulated_acquisition import (
    SimulatedInstrument,
    create_default_image,
)

logger = logging.getLogger(__name__)


class InstrumentWorker:
    """Blocking ZMQ worker that owns the simulated instrument instance.

    Parameters
    ----------
    instrument
        Simulated instrument object. This object lives only in the worker
        process.
    """

    def __init__(self, instrument: SimulatedInstrument) -> None:
        self.instrument = instrument
        self.handlers: dict[str, Callable[..., Any]] = {
            "health": self.health,
            "get_state": self.instrument.get_state,
            "set_blur": self.instrument.set_blur,
            "set_offset": self.instrument.set_offset,
            "set_config": self.instrument.set_config,
            "set_attribute": self.instrument.set_attribute,
            "set_parameters": self.instrument.set_parameters,
            "acquire_image": self.instrument.acquire_image,
            "acquire_line_scan": self.instrument.acquire_line_scan,
            "dump_array": self.instrument.dump_array,
            "get_attribute_payload": self.instrument.get_attribute_payload,
        }

    def health(self) -> dict[str, str]:
        """Return worker health."""
        return {"status": "ok"}

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Execute one command message and return a response envelope."""
        command_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(method, str) or method not in self.handlers:
            return error_response(command_id, f"Unknown method: {method}")
        if not isinstance(params, dict):
            return error_response(command_id, "Command params must be a JSON object.")
        try:
            result = self.handlers[method](**params)
            return ok_response(command_id, result)
        except Exception as exc:
            logger.exception("Worker command failed: %s", method)
            return error_response(command_id, str(exc))

    def serve(self, bind: str) -> None:
        """Serve worker commands forever on a ZMQ REP socket."""
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(bind)
        logger.info("Instrument worker listening on %s", bind)
        try:
            while True:
                raw_message = socket.recv_string()
                try:
                    message = json.loads(raw_message)
                    if not isinstance(message, dict):
                        response = error_response(None, "Command must be a JSON object.")
                    else:
                        response = self.handle_message(message)
                except json.JSONDecodeError as exc:
                    response = error_response(None, f"Invalid JSON command: {exc}")
                socket.send_string(json.dumps(response, allow_nan=True))
        finally:
            socket.close(linger=0)


def load_image(path: str | None) -> np.ndarray:
    """Load a source image from ``.npy`` or TIFF, or create the default image."""
    if path is None:
        return create_default_image()
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        image = np.load(image_path)
    elif suffix in {".tif", ".tiff"}:
        image = tifffile.imread(image_path)
    else:
        raise ValueError("Only .npy, .tif, and .tiff source images are supported.")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.float32)


def resolve_parameter_ranges(value: Any) -> list[tuple[float, ...]]:
    """Convert configured parameter ranges into tuple rows."""
    if value is None:
        return [(-1.0,), (1.0,)]
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("parameter_ranges must contain lower and upper bound rows.")
    return [tuple(float(item) for item in row) for row in value]


def build_instrument(args: argparse.Namespace) -> SimulatedInstrument:
    """Build the simulated instrument from CLI arguments."""
    config = load_config(args.config)
    parameter_config = get_config_section(config, "parameter_setting")

    return SimulatedInstrument(
        whole_image=load_image(resolve_setting(config, "acquisition", "image", args.image, None)),
        output_dir=Path(
            resolve_setting(
                config,
                "worker",
                "output_dir",
                args.output_dir,
                ".tmp/control_suite_mcp_dummy",
            )
        ),
        add_axis_ticks=resolve_setting(
            config,
            "acquisition",
            "add_axis_ticks",
            args.add_axis_ticks,
            False,
        ),
        n_ticks=resolve_setting(config, "acquisition", "n_ticks", args.n_ticks, 10),
        add_grid_lines=resolve_setting(
            config,
            "acquisition",
            "add_grid_lines",
            args.add_grid_lines,
            False,
        ),
        invert_yaxis=resolve_setting(
            config,
            "acquisition",
            "invert_yaxis",
            args.invert_yaxis,
            False,
        ),
        line_scan_gaussian_fit_y_threshold=resolve_setting(
            config,
            "acquisition",
            "line_scan_gaussian_fit_y_threshold",
            args.line_scan_gaussian_fit_y_threshold,
            0.0,
        ),
        plot_image_in_log_scale=resolve_setting(
            config,
            "acquisition",
            "plot_image_in_log_scale",
            args.plot_image_in_log_scale,
            False,
        ),
        line_scan_return_gaussian_fit=resolve_setting(
            config,
            "acquisition",
            "line_scan_return_gaussian_fit",
            args.line_scan_return_gaussian_fit,
            False,
        ),
        poisson_noise_scale=resolve_setting(
            config,
            "acquisition",
            "poisson_noise_scale",
            args.poisson_noise_scale,
            None,
        ),
        gaussian_psf_sigma=resolve_setting(
            config,
            "acquisition",
            "gaussian_psf_sigma",
            args.gaussian_psf_sigma,
            None,
        ),
        scan_jitter=resolve_setting(config, "acquisition", "scan_jitter", args.scan_jitter, None),
        add_line_scan_candidates_to_image=resolve_setting(
            config,
            "acquisition",
            "add_line_scan_candidates_to_image",
            args.add_line_scan_candidates_to_image,
            False,
        ),
        parameter_names=parameter_config.get("parameter_names", ["focus"]),
        true_parameters=parameter_config.get("true_parameters", [0.0]),
        parameter_ranges=resolve_parameter_ranges(parameter_config.get("parameter_ranges")),
        drift_factor=parameter_config.get("drift_factor", 0.0),
        blur_factor=parameter_config.get("blur_factor", 0.0),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the worker CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default=None)
    parser.add_argument("--config", default=None, help="Optional YAML worker configuration.")
    parser.add_argument("--image", default=None, help="Optional .npy source image.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--add-axis-ticks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--n-ticks", type=int, default=None)
    parser.add_argument("--add-grid-lines", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--invert-yaxis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--line-scan-gaussian-fit-y-threshold", type=float, default=None)
    parser.add_argument("--plot-image-in-log-scale", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--line-scan-return-gaussian-fit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--poisson-noise-scale", type=float, default=None)
    parser.add_argument("--gaussian-psf-sigma", type=float, default=None)
    parser.add_argument("--scan-jitter", type=float, default=None)
    parser.add_argument(
        "--add-line-scan-candidates-to-image",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main() -> None:
    """Run the instrument worker process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    config = load_config(args.config)
    bind = resolve_setting(config, "worker", "endpoint", args.bind, "tcp://127.0.0.1:5555")
    worker = InstrumentWorker(build_instrument(args))
    worker.serve(bind)


if __name__ == "__main__":
    main()
