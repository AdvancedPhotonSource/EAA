# Experiment Automation Agents (EAA)

## Table of Contents

- [Installation](#installation)
  - [Option 1: install via uv](#option-1-install-via-uv-recommended)
  - [Option 2: install via pip](#option-2-install-via-pip)
- [Quickstart guide](#quickstart-guide)
- [WebUI](#webui)
- [Model context protocol (MCP)](#model-context-protocol-mcp)


## Installation

### Option 1: install via uv (recommended)

uv is a Python environment and package manager that is fast and dependency-deterministic, 
offering better reproducibility guarantee. Unlike a conda environment, a uv
virtual environment is installed at the root directory of a project instead of
in a centralized location, making it more portable. 

If you haven't installed uv yet, follow the [official documentation](https://docs.astral.sh/uv/#installation) to install it.

Then clone the repostiroy, CD into the respository's root, and install the
package and its dependencies using 
```
uv sync
```
To install extra dependencies, which are usually needed by some specific tools used by
certain beamlines, use the `--extra` flag:
```
uv sync --extra name_of_extra_dependency_set
```
You can find the names of extra dependency sets in the `[project.optional-dependencies]`
section of `pyproject.toml`.

`uv sync` creates an environment in `/path/to/eaa/.venv`. 
To activate it, do
```
source .venv/bin/activate
```
If you are running a script located inside the repository, you can also directly
do it with `uv run` without activating the environment first:
```
uv run myscript.py
```

### Option 2: install via pip

First, create a conda environment with Python 3.11:
```
conda create -n eaa python=3.11
conda activate eaa
```

Then clone the respository to your hard drive. CD into the repository's 
root, and install it with
```
pip install -e .
```
The `-e` flag allows any changes made in the source code to immediately 
take effect without reinstallation when you import the package in Python.

To install extra dependencies, which are usually needed by some specific tools used by
certain beamlines, add the name of the extra dependency set in the command:
```
pip install -e .[name_of_extra_dependency_set]
```
You can find the names of extra dependency sets in the `[project.optional-dependencies]`
section of `pyproject.toml`.


## Quickstart guide

First, choose a task manager that contains the workflow you need. In this example,
we use `FeatureTrackingTaskManager` for a field-of-view search task.
```
from eaa.task_managers.imaging.feature_tracking import FeatureTrackingTaskManager
from eaa.api.llm_config import OpenAIConfig
```

This task manager needs an image acquisition tool. We use a simulated one:
```
from eaa.tools.imaging.acquisition import SimulatedAcquireImage
acquisition_tool = SimulatedAcquireImage(whole_image=<ndarray of simulation image>)
```

Create the task manager:
```
task_manager = FeatureTrackingTaskManager(
    llm_config=OpenAIConfig(
        model=<name of the model to use>,
        base_url=<base URL of the inference host>,
        api_key=<your API key>,
    ),
    tools=[acquisition_tool],
)
```
The model name, base URL and API key should be provided by the LLM provider.
The type of the object passed to `llm_config` determines the API to use. For
most LLM providers that offer an OpenAI-compatible API, `OpenAIConfig` will work.
AskSage is also supported through `AskSageConfig`, but there are currently some
limitations in the support.

With the task manager created, you can either run the workflow defined in the logic:
```
task_manager.run_fov_search(
    feature_description="the center of a Siemens star",
    y_range=(0, 600),
    x_range=(0, 600),
    fov_size=(200, 200),
    step_size=(200, 200),
)
```
or just start a chat with the agent:
```
task_manager.run_conversation()
```
The tool will be available during the chat, so you can still instruct it to perform
certain experiment tasks during the chat. 

To add an image to a message during the chat, append the image path to your message
as `<img path/to/img.png>`.

## WebUI

EAA has a webUI built with Chainlit. The webUI runs in a separate process,
and communicate with the agent process through a SQL database. Agent messages
are written into the database, which is polled by the webUI process and displayed;
user inputs in the webUI is also written into the database and read in the
agent process. 

To use this feature, specify the path of the SQL database to append or create
when creating the task manager by adding the following argument:
```
TaskManager(
    ...
    message_db_path="messages.db"
)
```
Then create a Python script `start_webui.py` with just the following two lines:
```
from eaa.gui.chat import *
set_message_db_path("messages.db")
```
Launch the WebUI using
```
chainlit run start_webui.py
```

### Auto-scrolling

Since the WebUI polls for messages from a SQL database, the native feature of
auto-scrolling (where the chat window automatically scrolls to the bottom to
show the latest message when a new message is received) does not work. To bring
auto-scrolling back, copy everything under `examples/webui/` (including the hidden
folder `.config`) into the working directory where `start_webui.py` is located.
Now the JS scroller will be injected into the WebUI to enable auto-scrolling.

## Model context protocol (MCP)

### MCP tool wrapper

EAA's MCP tool wrapper allows you to convert any tools that are subclasses of
`BaseTool` into an MCP tool and launch an MCP server offering these tools. 
This allows you to use the tools in EAA with other MCP clients such as
Claude Code and Gemini CLI.

We will illustrate how an MCP server can be set up using a simple example. A
calculator tool, subclassing `BaseTool`, is created in 
`src/eaa/tools/example_calculator.py`. To turn it into an MCP server, we
use `eaa.mcp.run_mcp_server_from_tools`. See `examples/mcp_calculator_server.py`
for an example.

After the server script is created, add it to the config JSON of your MCP client.
Refer to the documentations of the client on where this config file is located.
```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["path/to/mcp_calculator_server.py"]
    }
  }
}
```
If EAA is installed in a virtual environment, you will need to ask the MCP client
to activate the environment before launching the tool. Below is an example:
```json
{
  "mcpServers": {
    "calculator": {
      "command": "bash",
      "args": [
        "-c",
        "source /path/to/.venv/bin/activate && python path/to/mcp_calculator_server.py"
      ]
    }
  }
}
```
Now the MCP client should be able to run and connect to the MCP server and use the
tool.

### Using MCP tools (experimental)

EAA itself can also use MCP tools. While we still recommend using the built-in
`BaseTool` classes as function-calling tools if possible, using external MCP 
tools allows you to extend the agent's capability beyond what's in the built-in tools.

To use an external MCP tool, first create a config dictionary. This dictionary should
follow the [FastMCP format](https://gofastmcp.com/clients/client#configuration-format), 
which is the same format as the `settings.json` files used by many MCP clients such as
Claude, Gemini CLI and Cursor. The dictionary should be wrapped in an `MCPTool` object.
The object should then be passed to the task manager in the same way as other `BaseTool`
objects.

```python
from eaa.tools.mcp import MCPTool

config = {
    "mcpServers": {
        "image_acquisition": {
            "command": "python", 
            "args": ["./image_acquisition_mcp_server.py"]
        }
    }
}

mcp_tool = MCPTool(config)
```

Known issues:
- EAA currently cannot tell if an MCP tool returns an image path, and as such,
  routines in task managers that handle images will not work properly. 
- The MCP server restarts every time a query is made, resulting in additional
  overhead and loss of internal state. We are working on finding a way to keep
  the MCP client connection alive across queries. 
