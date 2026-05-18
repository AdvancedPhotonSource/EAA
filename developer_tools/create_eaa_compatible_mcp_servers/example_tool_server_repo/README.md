# Control Suite MCP Dummy

This project demonstrates the target deployment pattern:

```text
MCP client -> MCP over HTTP -> FastMCP server -> ZMQ -> instrument worker
```

The instrument worker recreates a simulated image-acquisition instrument. It
owns the instrument state and executes commands through a blocking ZMQ `REP`
loop. The MCP server exposes FastMCP tools and forwards each call to the worker
over a local ZMQ request.

The two-process design is intentional. FastMCP and many instrument-control
libraries may each depend on their own event loop, callback thread, or runtime
lifecycle. Running the MCP HTTP server and the instrument controller in
separate OS processes prevents those async runtimes from sharing a Python event
loop or thread. ZMQ is used as a narrow local command boundary between the two
processes.

In this pattern:

- The FastMCP server owns MCP/HTTP request handling.
- The instrument worker owns the control library, instrument state, and command
  execution.
- ZMQ carries JSON-serializable commands and responses between them.
- The worker can keep a simple blocking command loop or adapt internally to the
  event loop required by a real instrument library.
- The launcher starts both processes for convenience, but it does not merge
  their runtimes into one process.

## Setup

```bash
uv sync
```

Copy and edit the example configuration before connecting to real hardware or
changing artifact paths:

```bash
cp configs/simulated_parameter_tuning.yaml configs/local.yaml
```

## Run

Start both processes with one supervisor command:

```bash
uv run control-suite \
  --config configs/simulated_parameter_tuning.yaml \
  --worker-endpoint tcp://127.0.0.1:5555 \
  --worker-startup-timeout-s 10 \
  --request-timeout-ms 30000 \
  --mcp-host 0.0.0.0 \
  --mcp-port 8050 \
  --mcp-path /mcp
```

Every console script accepts `--config PATH`. Explicit CLI flags override YAML
values, which lets operators make temporary runtime changes without editing the
file.

To use only YAML defaults, run:

```bash
uv run control-suite --config configs/simulated_parameter_tuning.yaml
```

The YAML `acquisition.image` setting accepts `.npy`, `.tif`, and `.tiff`
source images. If it is `null`, the worker uses the built-in synthetic image.

The launcher starts the worker first, waits for a ZMQ health response, then
starts the MCP server. The worker and MCP server remain separate OS processes.

You can also start each process manually.

Start the worker:

```bash
uv run control-suite-worker \
  --config configs/simulated_parameter_tuning.yaml \
  --bind tcp://127.0.0.1:5555
```

Start the MCP HTTP server in another terminal:

```bash
uv run control-suite-mcp \
  --config configs/simulated_parameter_tuning.yaml \
  --worker tcp://127.0.0.1:5555 \
  --timeout-ms 30000 \
  --host 0.0.0.0 \
  --port 8050 \
  --path /mcp
```

Any MCP client that supports HTTP transport can connect to:

```text
http://127.0.0.1:8050/mcp
```

## Client Configuration

Many MCP clients use a JSON configuration with an `mcpServers` table. For this
server, the client entry should point to the HTTP MCP endpoint:

```json
{
  "mcpServers": {
    "control_suite_dummy": {
      "url": "http://127.0.0.1:8050/mcp",
      "transport": "http"
    }
  }
}
```

Codex uses TOML in `~/.codex/config.toml`. Add this entry:

```toml
[mcp_servers.control-suite-dummy]
url = "http://127.0.0.1:8050/mcp"
```

The equivalent Codex CLI command is:

```bash
codex mcp add control-suite-dummy --url http://127.0.0.1:8050/mcp
```

## Tools

- `acquire_image(x_center, y_center, size_y, size_x, scan_step=1.0)`
- `acquire_line_scan(x_center, y_center, length, scan_step, angle=0.0)`
- `dump_array(buffer_name)`
- `set_blur(blur)`
- `set_offset(y_offset, x_offset)`
- `set_config(name, value)`
- `set_attribute(name, value)`
- `set_parameters(parameters)`
- `get_attribute_payload(name)`
- `get_state()`
- `health()`

`acquire_image` returns `img_path` and `psize`. It also updates the worker-side
`image_0`, `image_km1`, and `image_k` buffers. `dump_array` accepts those
buffer names and writes the selected buffer as a `.npy` artifact, returning the
artifact `path` instead of embedding large arrays in JSON. `acquire_line_scan`
returns `img_path` and numeric `fwhm`; when `line_scan_return_gaussian_fit` is
enabled it also returns fit metadata.

## Configuration

The example YAML contains these sections:

- `launcher`: worker startup timeout and supervisor settings.
- `worker`: ZMQ endpoint, request timeout, output directory, and safety flags.
- `mcp`: HTTP host, port, and path.
- `acquisition`: image source, plotting options, line-scan fit options, noise,
  jitter, and point-spread simulation.
- `parameter_setting`: parameter names, ranges, true values, and simulated
  blur/drift factors.

Generated PNG and `.npy` artifacts are written below `worker.output_dir`. Use a
directory that is readable by the MCP client and by any EAA process that will
consume returned artifact paths.
