import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any
from urllib import error, parse, request

from eaa_core.tool.base import BaseTool, ExposedToolSpec


class HTTPTransportedTool(BaseTool):
    """Expose one or more remote HTTP POST endpoints through the tool interface.

    Notes
    -----
    ``tool_definitions`` is a sequence of dictionaries. Each dictionary defines
    one tool exposed to the agent and must include:

    - ``tool_name``: tool name exposed to the agent
    - ``endpoint_name``: endpoint path appended to ``base_url``
    - ``description``: human-readable tool description
    - ``input_signautures`` or ``input_signatures``: input schema definition
    - ``output_signatures``: output schema definition

    The input and output signature values may be provided in either of two
    forms:

    1. A complete JSON-schema-like object, for example::

           {
               "type": "object",
               "properties": {
                   "query": {
                       "type": "string",
                       "description": "Search phrase.",
                   },
                   "limit": {
                       "type": "integer",
                       "description": "Maximum number of matches.",
                   },
               },
               "required": ["query"],
           }

    2. A shorthand mapping from field names to field definitions, for example::

           {
               "query": {
                   "type": "string",
                   "description": "Search phrase.",
                   "required": True,
               },
               "limit": {
                   "type": "integer",
                   "description": "Maximum number of matches.",
               },
           }

       In the shorthand form, each field value may also be a plain string,
       which is treated as a string-typed field description.

    Supported field ``type`` values are the JSON-schema primitives understood
    by the local signature builder: ``string``, ``integer``, ``number``,
    ``boolean``, ``array``, and ``object``. Nullable forms such as
    ``["string", "null"]`` are also accepted. If a field uses an unknown
    ``type``, the schema is still passed through, but local introspection falls
    back to ``Any`` for that field. Dictionary-shaped inputs or outputs should
    use ``{"type": "object"}``, typically with nested ``properties`` when the
    structure is known.

    The OpenAI tool schema uses the compiled input signature as
    ``function.parameters``. The compiled output signature is not enforced
    during parsing; it is attached to the tool description so the model can see
    the expected result shape.
    """

    def __init__(
        self,
        base_url: str,
        tool_definitions: Sequence[Mapping[str, Any]],
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        request_payload_builder: Callable[[dict[str, Any]], Any] | None = None,
        response_parser: Callable[[Any], Any] | None = None,
        require_approval: bool = False,
        build: bool = True,
        name: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the HTTP tool wrapper.

        Parameters
        ----------
        base_url : str
            Base URL used to construct each tool endpoint URL.
        tool_definitions : Sequence[Mapping[str, Any]]
            Sequence of per-tool definitions. Each item must provide
            ``tool_name``, ``endpoint_name``, ``description``,
            ``input_signautures``, and ``output_signatures``. Supported field
            ``type`` values are
            ``string``, ``integer``, ``number``, ``boolean``, ``array``, and
            ``object``. Nullable forms such as ``["string", "null"]`` are also
            accepted. Use ``"object"`` for dictionary-shaped values.
        headers : Mapping[str, str], optional
            Extra HTTP headers to include with POST requests.
        timeout : float | None, optional
            Request timeout in seconds. When ``None``, no per-request timeout is
            enforced by this wrapper.
        request_payload_builder : Callable[[dict[str, Any]], Any], optional
            Callable that builds the outbound POST payload from the tool
            arguments. By default, the client sends the tool arguments directly
            as the JSON request body.
        response_parser : Callable[[Any], Any], optional
            Callable that converts the decoded HTTP response into the final tool
            result. The default returns ``response["result"]`` when available,
            otherwise the decoded payload itself.
        require_approval : bool, optional
            Whether execution should require approval.
        build : bool, optional
            Whether to build immediately.
        name : str, optional
            Instance-level tool collection name override.
        *args
            Positional arguments forwarded to ``BaseTool``.
        **kwargs
            Keyword arguments forwarded to ``BaseTool``.
        """
        self.base_url = base_url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.request_payload_builder = request_payload_builder
        self.response_parser = response_parser
        self.tool_definitions = self.normalize_tool_definitions(tool_definitions)
        self.tool_definitions_by_name = {
            definition["tool_name"]: definition for definition in self.tool_definitions
        }
        super().__init__(
            build=build,
            require_approval=require_approval,
            name=name,
            *args,
            **kwargs,
        )

    def build(self, *args, **kwargs) -> None:
        """Build internal state for the HTTP wrapper."""

    @classmethod
    def normalize_tool_definitions(
        cls,
        tool_definitions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize user-supplied tool definitions.

        Parameters
        ----------
        tool_definitions : Sequence[Mapping[str, Any]]
            Raw tool definitions supplied to the constructor.

        Returns
        -------
        list[dict[str, Any]]
            Normalized tool definitions with compiled schemas.

        Raises
        ------
        TypeError
            If ``tool_definitions`` is not a sequence of mappings.
        ValueError
            If required fields are missing or tool names are duplicated.
        """
        if not isinstance(tool_definitions, Sequence) or isinstance(tool_definitions, (str, bytes)):
            raise TypeError("tool_definitions must be a sequence of mappings.")

        normalized_definitions: list[dict[str, Any]] = []
        seen_tool_names: set[str] = set()
        for definition in tool_definitions:
            if not isinstance(definition, Mapping):
                raise TypeError("Each tool definition must be a mapping.")

            tool_name = definition.get("tool_name")
            endpoint_name = definition.get("endpoint_name")
            description = definition.get("description")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("Each tool definition must include a non-empty string tool_name.")
            if tool_name in seen_tool_names:
                raise ValueError(f"Duplicate tool_name in tool_definitions: {tool_name}")
            if not isinstance(endpoint_name, str) or not endpoint_name.strip():
                raise ValueError(
                    f"Tool definition '{tool_name}' must include a non-empty string endpoint_name."
                )
            if not isinstance(description, str):
                raise ValueError(
                    f"Tool definition '{tool_name}' must include a string description."
                )

            if "input_signautures" not in definition:
                raise ValueError(
                    f"Tool definition '{tool_name}' must include input_signautures."
                )
            if "output_signatures" not in definition:
                raise ValueError(
                    f"Tool definition '{tool_name}' must include output_signatures."
                )

            normalized_definitions.append(
                {
                    "tool_name": tool_name,
                    "endpoint_name": endpoint_name,
                    "description": description,
                    "input_schema": cls.compile_schema(definition.get("input_signautures")),
                    "output_schema": cls.compile_schema(definition.get("output_signatures")),
                }
            )
            seen_tool_names.add(tool_name)

        return normalized_definitions

    @staticmethod
    def compile_schema(signature_description: Mapping[str, Any] | None) -> dict[str, Any]:
        """Compile a structured signature description into a JSON schema.

        Parameters
        ----------
        signature_description : Mapping[str, Any], optional
            Accepted input shapes are:

            - ``None``, which compiles to an empty object schema.
            - A JSON-schema-like mapping that already contains top-level
              ``type`` or ``properties`` keys.
            - A shorthand field mapping, where each key is a field name and
              each value is one of:

              - a mapping containing field schema keys such as ``type``,
                ``description``, and ``required``
              - a string description, which compiles to a string field

            In the shorthand field mapping form, ``required=True`` on a field
            is moved into the top-level ``required`` list. Supported field
            ``type`` values are ``string``, ``integer``, ``number``,
            ``boolean``, ``array``, and ``object``. Nullable forms such as
            ``["string", "null"]`` are also accepted. Use ``"object"`` for
            dictionary-shaped values.

        Returns
        -------
        dict[str, Any]
            Normalized JSON schema object.
        """
        if signature_description is None:
            return {
                "type": "object",
                "properties": {},
                "required": [],
            }

        schema = dict(signature_description)
        if "type" in schema or "properties" in schema:
            compiled = dict(schema)
            compiled.setdefault("type", "object")
            compiled.setdefault("properties", {})
            compiled["properties"] = {
                key: dict(value) if isinstance(value, Mapping) else {"type": "string"}
                for key, value in dict(compiled["properties"]).items()
            }
            compiled.setdefault("required", [])
            return compiled

        properties: dict[str, Any] = {}
        required: list[str] = []
        for field_name, field_description in schema.items():
            if isinstance(field_description, Mapping):
                field_schema = dict(field_description)
            elif isinstance(field_description, str):
                field_schema = {
                    "type": "string",
                    "description": field_description,
                }
            else:
                field_schema = {"type": "string"}
            if field_schema.pop("required", False):
                required.append(field_name)
            properties[field_name] = field_schema

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def build_openai_schema(self, tool_definition: Mapping[str, Any]) -> dict[str, Any]:
        """Build the OpenAI-compatible schema for one exposed tool.

        Parameters
        ----------
        tool_definition : Mapping[str, Any]
            Normalized tool definition.

        Returns
        -------
        dict[str, Any]
            Function-calling schema describing the remote tool.
        """
        return {
            "type": "function",
            "function": {
                "name": tool_definition["tool_name"],
                "description": self.build_tool_description(tool_definition),
                "parameters": tool_definition["input_schema"],
            },
        }

    def build_tool_description(self, tool_definition: Mapping[str, Any]) -> str:
        """Build the description presented to the model.

        Parameters
        ----------
        tool_definition : Mapping[str, Any]
            Normalized tool definition.

        Returns
        -------
        str
            Description enriched with output schema information when provided.
        """
        output_schema = tool_definition["output_schema"]
        if not output_schema.get("properties"):
            return tool_definition["description"]
        output_schema_json = json.dumps(output_schema, sort_keys=True)
        return f'{tool_definition["description"]}\n\nOutput schema: {output_schema_json}'

    @staticmethod
    def python_type_from_json_schema(schema: dict[str, Any]) -> type[Any] | Any:
        """Map a JSON schema field type to a Python annotation.

        Parameters
        ----------
        schema : dict[str, Any]
            JSON schema fragment for one field.

        Returns
        -------
        type[Any] | Any
            Best-effort Python type.
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

    def annotation_from_json_schema(self, schema: dict[str, Any]) -> Any:
        """Build a Python annotation from a JSON schema field.

        Parameters
        ----------
        schema : dict[str, Any]
            JSON schema fragment for one field.

        Returns
        -------
        Any
            Annotation suitable for synthetic signatures.
        """
        annotation = self.python_type_from_json_schema(schema)
        description = schema.get("description")
        if isinstance(description, str) and description:
            return Annotated[annotation, description]
        return annotation

    def build_signature_from_input_schema(
        self,
        input_schema: dict[str, Any],
    ) -> inspect.Signature:
        """Construct a keyword-only Python signature from the input schema.

        Parameters
        ----------
        input_schema : dict[str, Any]
            Input JSON schema for the remote tool.

        Returns
        -------
        inspect.Signature
            Synthetic signature for local introspection.
        """
        parameters = []
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        for field_name, field_schema in properties.items():
            normalized_schema = field_schema if isinstance(field_schema, dict) else {}
            default = inspect.Parameter.empty if field_name in required else None
            parameters.append(
                inspect.Parameter(
                    name=field_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=self.annotation_from_json_schema(normalized_schema),
                )
            )
        return inspect.Signature(parameters=parameters)

    def default_request_payload_builder(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the default outbound POST payload.

        Parameters
        ----------
        arguments : dict[str, Any]
            Arguments passed by the local caller.

        Returns
        -------
        dict[str, Any]
            Default JSON payload.
        """
        return dict(arguments)

    def default_response_parser(self, response_payload: Any) -> Any:
        """Parse the default response format.

        Parameters
        ----------
        response_payload : Any
            Decoded HTTP response.

        Returns
        -------
        Any
            Parsed tool result.
        """
        if isinstance(response_payload, dict) and "result" in response_payload:
            return response_payload["result"]
        return response_payload

    def decode_response_body(self, body: bytes, content_type: str) -> Any:
        """Decode the HTTP response body.

        Parameters
        ----------
        body : bytes
            Raw response payload.
        content_type : str
            Content-Type header value.

        Returns
        -------
        Any
            JSON-decoded payload when possible, otherwise text.
        """
        if not body:
            return None
        text = body.decode("utf-8", errors="replace")
        if "json" in content_type.lower():
            return json.loads(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def build_endpoint_url(self, endpoint_name: str) -> str:
        """Construct the full URL for a tool endpoint.

        Parameters
        ----------
        endpoint_name : str
            Endpoint path relative to ``base_url``.

        Returns
        -------
        str
            Absolute endpoint URL.
        """
        return parse.urljoin(f"{self.base_url.rstrip('/')}/", endpoint_name.lstrip("/"))

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke the selected remote HTTP endpoint and return the parsed result.

        Parameters
        ----------
        tool_name : str
            Tool name exposed to the agent.
        arguments : dict[str, Any]
            Tool arguments supplied by the agent.

        Returns
        -------
        Any
            Parsed tool response.
        """
        tool_definition = self.tool_definitions_by_name.get(tool_name)
        if tool_definition is None:
            raise ValueError(f"Unknown HTTP transported tool: {tool_name}")

        payload_builder = self.request_payload_builder or self.default_request_payload_builder
        payload = payload_builder(arguments)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self.headers,
        }
        http_request = request.Request(
            self.build_endpoint_url(tool_definition["endpoint_name"]),
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                body = response.read()
                decoded = self.decode_response_body(
                    body=body,
                    content_type=response.headers.get("Content-Type", ""),
                )
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP tool call failed with status {exc.code}: {error_body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"HTTP tool call failed: {exc.reason}") from exc

        parser = self.response_parser or self.default_response_parser
        return parser(decoded)

    def make_tool_callable(self, tool_definition: Mapping[str, Any]) -> Callable[..., Any]:
        """Create the local callable that proxies one HTTP endpoint.

        Parameters
        ----------
        tool_definition : Mapping[str, Any]
            Normalized tool definition.

        Returns
        -------
        Callable[..., Any]
            Synchronous wrapper used by the agent tool executor.
        """
        tool_name = tool_definition["tool_name"]

        def runner(**kwargs: Any) -> Any:
            return self.call_tool(tool_name=tool_name, arguments=kwargs)

        runner.__name__ = tool_name
        runner.__doc__ = self.build_tool_description(tool_definition)
        runner.__signature__ = self.build_signature_from_input_schema(
            tool_definition["input_schema"]
        )
        runner.__annotations__ = {
            parameter_name: parameter.annotation
            for parameter_name, parameter in runner.__signature__.parameters.items()
        }
        runner.__annotations__["return"] = Any
        return runner

    def discover_tools(self) -> list[ExposedToolSpec]:
        """Expose all configured HTTP endpoints as independent tools.

        Returns
        -------
        list[ExposedToolSpec]
            Exposed tool specifications, one per tool definition.
        """
        return [
            ExposedToolSpec(
                name=tool_definition["tool_name"],
                function=self.make_tool_callable(tool_definition),
                require_approval=self.require_approval,
                schema=self.build_openai_schema(tool_definition),
            )
            for tool_definition in self.tool_definitions
        ]

    def get_all_schema(self) -> list[dict[str, Any]]:
        """Return the schemas for all exposed HTTP tools.

        Returns
        -------
        list[dict[str, Any]]
            Schema list for the configured tools.
        """
        return [spec.schema for spec in self.exposed_tools if spec.schema is not None]

    def get_all_tool_names(self) -> list[str]:
        """Return the exposed tool names.

        Returns
        -------
        list[str]
            Tool name list.
        """
        return [spec.name for spec in self.exposed_tools]
