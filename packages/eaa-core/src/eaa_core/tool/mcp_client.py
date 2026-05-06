import asyncio
import inspect
from typing import Annotated, Any

import fastmcp

from eaa_core.tool.base import ExposedToolSpec, BaseTool


class MCPTool(BaseTool):
    """Expose external MCP tools through the normal ``BaseTool`` interface."""

    def __init__(
        self,
        config: dict,
        require_approval: bool = False,
        build: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the MCP tool wrapper.

        Parameters
        ----------
        config : dict
            FastMCP-compatible client configuration.
        require_approval : bool, optional
            Whether each remote tool invocation should require approval.
        build : bool, optional
            Whether to connect and discover remote tools immediately.
        *args
            Positional arguments forwarded to ``BaseTool``.
        **kwargs
            Keyword arguments forwarded to ``BaseTool``.
        """
        self.config = config
        self._client = None
        self._connected = False
        self._loop = asyncio.new_event_loop()
        super().__init__(
            build=build,
            require_approval=require_approval,
            *args,
            **kwargs,
        )

    def build(self, *args, **kwargs) -> None:
        """Build the MCP client wrapper.

        Parameters
        ----------
        *args
            Unused positional compatibility arguments.
        **kwargs
            Unused keyword compatibility arguments.
        """

    def _run_coroutine(self, coroutine):
        """Run an async MCP client coroutine on the dedicated event loop.

        Parameters
        ----------
        coroutine : coroutine
            Coroutine to execute.

        Returns
        -------
        Any
            Coroutine result.
        """
        return self._loop.run_until_complete(coroutine)

    async def _ensure_connected(self) -> None:
        """Ensure the MCP client is connected.

        Returns
        -------
        None
        """
        if not self._connected or self._client is None:
            await self.connect()

    async def connect(self) -> None:
        """Connect to the configured MCP server set.

        Returns
        -------
        None
        """
        if self._client is not None:
            await self.disconnect()
        self._client = fastmcp.Client(self.config)
        await self._client.__aenter__()
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the configured MCP server set.

        Returns
        -------
        None
        """
        if self._client is not None and self._connected:
            await self._client.__aexit__(None, None, None)
            self._connected = False
            self._client = None

    async def list_tools(self):
        """List the remote MCP tools.

        Returns
        -------
        list
            MCP tool descriptors returned by FastMCP.
        """
        await self._ensure_connected()
        return await self._client.list_tools()

    async def list_resources(self):
        """List the remote MCP resources.

        Returns
        -------
        list
            MCP resources returned by FastMCP.
        """
        await self._ensure_connected()
        return await self._client.list_resources()

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server.

        Parameters
        ----------
        tool_name : str
            Name of the remote tool.
        arguments : dict
            Tool arguments to forward to the MCP server.

        Returns
        -------
        Any
            Structured tool result when available, otherwise the unstructured
            response content.
        """
        await self._ensure_connected()
        result = await self._client.call_tool(tool_name, arguments)
        structured_content = getattr(result, "structured_content", None)
        if isinstance(structured_content, dict) and "result" in structured_content:
            return structured_content["result"]
        if structured_content is not None:
            return structured_content
        content = getattr(result, "content", None)
        if isinstance(content, list):
            texts = []
            for item in content:
                text = getattr(item, "text", None)
                if text is not None:
                    texts.append(text)
            if texts:
                return "\n".join(texts)
        return content

    def _build_openai_schema_from_mcp_tool(self, tool) -> dict[str, Any]:
        """Convert a FastMCP tool descriptor into an OpenAI-style schema.

        Parameters
        ----------
        tool
            MCP tool descriptor returned by ``list_tools``.

        Returns
        -------
        dict[str, Any]
            OpenAI-compatible function-call schema.
        """
        input_schema = dict(tool.inputSchema or {})
        input_schema.setdefault("type", "object")
        input_schema.setdefault("properties", {})
        input_schema.setdefault("required", [])
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": input_schema,
            },
        }

    def _python_type_from_json_schema(self, schema: dict[str, Any]) -> type[Any] | Any:
        """Map a JSON Schema field type to a Python type annotation.

        Parameters
        ----------
        schema : dict[str, Any]
            JSON Schema fragment describing a single parameter.

        Returns
        -------
        type[Any] | Any
            Best-effort Python type for introspection purposes.
        """
        json_type = schema.get("type")
        if isinstance(json_type, list):
            non_null_types = [value for value in json_type if value != "null"]
            if len(non_null_types) == 1:
                json_type = non_null_types[0]
            else:
                return Any
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, Any)

    def _annotation_from_json_schema(self, schema: dict[str, Any]) -> Any:
        """Build a Python annotation from a JSON Schema parameter definition.

        Parameters
        ----------
        schema : dict[str, Any]
            JSON Schema fragment describing a single parameter.

        Returns
        -------
        Any
            Python annotation suitable for an ``inspect.Signature``.
        """
        annotation = self._python_type_from_json_schema(schema)
        description = schema.get("description")
        if isinstance(description, str) and description:
            return Annotated[annotation, description]
        return annotation

    def _build_signature_from_input_schema(
        self,
        input_schema: dict[str, Any],
    ) -> inspect.Signature:
        """Construct a synthetic callable signature from MCP input schema.

        Parameters
        ----------
        input_schema : dict[str, Any]
            MCP tool input schema.

        Returns
        -------
        inspect.Signature
            Keyword-only signature matching the remote tool parameters.
        """
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        parameters = []
        for name, schema in properties.items():
            field_schema = schema if isinstance(schema, dict) else {}
            default = inspect.Parameter.empty if name in required else None
            parameters.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=self._annotation_from_json_schema(field_schema),
                )
            )
        return inspect.Signature(parameters=parameters)

    def _make_tool_callable(
        self,
        tool_name: str,
        input_schema: dict[str, Any],
        description: str = "",
    ):
        """Create a synchronous callable that proxies a remote MCP tool.

        Parameters
        ----------
        tool_name : str
            Name of the remote tool to proxy.
        input_schema : dict[str, Any]
            MCP input schema used to synthesize a Python signature.
        description : str, optional
            Human-readable remote tool description.

        Returns
        -------
        callable
            Synchronous wrapper used by the serial tool executor.
        """

        def runner(**kwargs):
            return self._run_coroutine(self.call_tool(tool_name, kwargs))

        runner.__name__ = tool_name
        runner.__doc__ = description
        runner.__signature__ = self._build_signature_from_input_schema(input_schema)
        runner.__annotations__ = {
            name: parameter.annotation
            for name, parameter in runner.__signature__.parameters.items()
        }
        runner.__annotations__["return"] = Any
        return runner

    def discover_tools(self) -> list[ExposedToolSpec]:
        """Discover remote MCP tools and expose them as EAA tools.

        Returns
        -------
        list[ExposedToolSpec]
            Remote tools exposed through the standard EAA tool interface.
        """
        remote_tools = self._run_coroutine(self.list_tools())
        exposed_tools = []
        for remote_tool in remote_tools:
            input_schema = dict(remote_tool.inputSchema or {})
            function = self._make_tool_callable(
                remote_tool.name,
                input_schema=input_schema,
                description=remote_tool.description or "",
            )
            if not hasattr(self, remote_tool.name):
                setattr(self, remote_tool.name, function)
            exposed_tools.append(
                ExposedToolSpec(
                    name=remote_tool.name,
                    function=function,
                    require_approval=self.require_approval,
                    schema=self._build_openai_schema_from_mcp_tool(remote_tool),
                )
            )
        return exposed_tools

    def get_all_schema(self) -> list[dict[str, Any]]:
        """Return OpenAI-style schemas for all remote tools.

        Returns
        -------
        list[dict[str, Any]]
            Model-facing schemas for the remote MCP tools.
        """
        return [spec.schema for spec in self.exposed_tools if spec.schema is not None]

    def get_all_tool_names(self) -> list[str]:
        """Return the names of all remote tools.

        Returns
        -------
        list[str]
            Remote MCP tool names.
        """
        return [spec.name for spec in self.exposed_tools]

    async def __aenter__(self):
        """Enter the async context manager.

        Returns
        -------
        MCPTool
            Connected MCP tool wrapper.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager.

        Parameters
        ----------
        exc_type
            Exception type, if any.
        exc_val
            Exception value, if any.
        exc_tb
            Exception traceback, if any.

        Returns
        -------
        None
        """
        await self.disconnect()

    def __del__(self):
        """Perform best-effort MCP client cleanup.

        Returns
        -------
        None
        """
        try:
            if self._connected and self._client is not None:
                self._loop.run_until_complete(self.disconnect())
            if not self._loop.is_closed():
                self._loop.close()
        except Exception:
            pass
