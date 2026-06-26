"""HTML/CSS/JavaScript WebUI for EAA task managers."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import urllib.error
import urllib.request
from importlib.resources import files
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from starlette.staticfiles import StaticFiles


class HTMLWebUIBase:
    """Browser chat UI backed by the agent-side WebUI runtime API.

    Parameters
    ----------
    runtime_url : str
        Base URL of the agent-side WebUI runtime API.
    title : str, default="EAA WebUI"
        Browser and header title.
    poll_interval : float, default=1.0
        Reconnect backoff interval in seconds.
    """

    image_route = "/api/image"
    events_route = "/api/events"
    state_route = "/api/state"
    send_route = "/api/input"
    interrupt_route = "/api/interrupt"
    approval_route = "/api/approval"
    upload_route = "/api/upload-image"
    skill_catalog_route = "/api/skill-catalog"
    tool_schema_route = "/api/tool-schemas"
    mathjax_route = "/static/mathjax"
    webui_static_route = "/static/webui"
    markdown_extras = ["fenced-code-blocks", "tables", "code-friendly", "break-on-newline"]
    foldable_roles = {"user", "user_webui", "tool"}
    folded_message_line_limit = 10

    def __init__(
        self,
        runtime_url: str = "http://127.0.0.1:8010",
        *,
        title: str = "EAA WebUI",
        poll_interval: float = 1.0,
    ) -> None:
        self.runtime_url = runtime_url.rstrip("/")
        self.title = title
        self.poll_interval = poll_interval

    def build_app(self) -> FastAPI:
        """Build the FastAPI application serving the WebUI shell."""
        app = FastAPI(title=self.title)
        mathjax_dir = files("eaa_core").joinpath("gui/static/mathjax")
        app.mount(
            self.mathjax_route,
            StaticFiles(directory=str(mathjax_dir)),
            name="eaa_mathjax",
        )
        webui_dir = self.webui_static_dir()
        if webui_dir.is_dir():
            app.mount(
                self.webui_static_route,
                StaticFiles(directory=str(webui_dir)),
                name="eaa_webui",
            )

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(self.page_html())

        @app.get(self.events_route)
        def events() -> StreamingResponse:
            return StreamingResponse(
                self.proxy_event_stream(),
                media_type="text/event-stream",
            )

        @app.api_route("/api/{runtime_path:path}", methods=["GET", "POST"])
        async def runtime_proxy(runtime_path: str, request: Request) -> Response:
            return await self.proxy_runtime_request(runtime_path, request)

        return app

    def runtime_api_url(self, runtime_path: str, query: str = "") -> str:
        """Return the runtime API URL for one proxied path."""
        url = f"{self.runtime_url}/api/{runtime_path}"
        if query:
            url = f"{url}?{query}"
        return url

    async def proxy_runtime_request(
        self,
        runtime_path: str,
        request: Request,
    ) -> Response:
        """Forward one browser API request to the agent runtime."""
        body = await request.body()
        url = self.runtime_api_url(runtime_path, request.url.query)
        headers = {}
        content_type = request.headers.get("content-type")
        if content_type:
            headers["Content-Type"] = content_type
        return await self.run_blocking_proxy_request(
            method=request.method,
            url=url,
            body=body or None,
            headers=headers,
        )

    async def run_blocking_proxy_request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> Response:
        """Run a standard-library HTTP proxy request off the event loop."""
        import asyncio

        return await asyncio.to_thread(
            self.blocking_proxy_request,
            method=method,
            url=url,
            body=body,
            headers=headers,
        )

    @staticmethod
    def response_headers(upstream_headers: Any) -> dict[str, str]:
        """Return response headers safe to pass through the proxy."""
        allowed = {
            "cache-control",
            "etag",
            "last-modified",
            "content-disposition",
        }
        return {
            key: value
            for key, value in upstream_headers.items()
            if key.lower() in allowed
        }

    def blocking_proxy_request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> Response:
        """Forward one non-streaming request to the runtime."""
        runtime_request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(runtime_request, timeout=30) as upstream:
                content = upstream.read()
                content_type = upstream.headers.get("content-type")
                return Response(
                    content=content,
                    status_code=upstream.status,
                    media_type=content_type,
                    headers=self.response_headers(upstream.headers),
                )
        except urllib.error.HTTPError as error:
            content = error.read()
            return Response(
                content=content,
                status_code=error.code,
                media_type=error.headers.get("content-type"),
                headers=self.response_headers(error.headers),
            )
        except urllib.error.URLError as error:
            return Response(
                content=json.dumps({"error": f"Runtime unavailable: {error.reason}"}),
                status_code=502,
                media_type="application/json",
            )

    def proxy_event_stream(self) -> Any:
        """Yield SSE bytes from the agent runtime."""
        runtime_request = urllib.request.Request(
            self.runtime_api_url("events"),
            headers={"Accept": "text/event-stream"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(runtime_request, timeout=None) as upstream:
                while True:
                    chunk = upstream.readline()
                    if not chunk:
                        break
                    yield chunk
        except urllib.error.URLError as error:
            payload = json.dumps({"error": f"Runtime unavailable: {error.reason}"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")

    def run(self, host: str = "127.0.0.1", port: int = 8008) -> None:
        """Run the WebUI server."""
        import uvicorn

        print(f"EAA WebUI running at http://{host}:{port} (runtime: {self.runtime_url})")
        uvicorn.run(self.build_app(), host=host, port=port)

    def webui_static_dir(self) -> Any:
        """Return the packaged Vite WebUI static directory."""
        return files("eaa_core").joinpath("gui/static/webui")

    def webui_index_html(self) -> str:
        """Return the built Vite WebUI index document."""
        index_path = self.webui_static_dir().joinpath("index.html")
        if not index_path.is_file():
            return self.missing_build_html()
        return index_path.read_text(encoding="utf-8")

    def page_config(self) -> dict[str, Any]:
        """Return browser bootstrap configuration for the React WebUI."""
        return {
            "title": self.title,
            "runtimeUrl": self.runtime_url,
            "pollIntervalMs": max(100, int(self.poll_interval * 1000)),
            "routes": {
                "events": self.events_route,
                "state": self.state_route,
                "image": self.image_route,
                "send": self.send_route,
                "interrupt": self.interrupt_route,
                "approval": self.approval_route,
                "upload": self.upload_route,
                "skillCatalog": self.skill_catalog_route,
                "toolSchemas": self.tool_schema_route,
                "mathjax": self.mathjax_route,
            },
        }

    def page_config_script(self) -> str:
        """Return the runtime configuration script injected before React loads."""
        return f"""<script>
    window.EAA_WEBUI_CONFIG = {json.dumps(self.page_config())};
  </script>
  <link rel=\"preload\" as=\"script\" href=\"{self.mathjax_route}/es5/tex-svg-full.js\">"""

    def missing_build_html(self) -> str:
        """Return a minimal page when packaged WebUI assets are missing."""
        return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(self.title)}</title>
</head>
<body>
  <main style=\"font-family: sans-serif; max-width: 720px; margin: 4rem auto; line-height: 1.5;\">
    <h1>{html.escape(self.title)}</h1>
    <p>The built React WebUI assets are missing.</p>
    <p>From <code>packages/eaa-core/webui</code>, run <code>npm install</code> and <code>npm run build</code>.</p>
  </main>
</body>
</html>"""

    def page_html(self) -> str:
        """Return the complete HTML document for the React WebUI."""
        document = self.webui_index_html()
        document = document.replace("<title>EAA WebUI</title>", f"<title>{html.escape(self.title)}</title>")
        config_script = self.page_config_script()
        if "<head>" in document:
            return document.replace("<head>", f"<head>\n  {config_script}", 1)
        if "</head>" in document:
            return document.replace("</head>", f"  {config_script}\n</head>", 1)
        return f"{config_script}\n{document}"

    def script(self) -> str:
        """Return an empty script placeholder retained for API compatibility."""
        return ""

    def styles(self) -> str:
        """Return an empty style placeholder retained for API compatibility."""
        return ""

def run_html_webui(
    runtime_url: str = "http://127.0.0.1:8010",
    *,
    host: str = "127.0.0.1",
    port: int = 8008,
    title: str = "EAA WebUI",
    poll_interval: float = 1.0,
) -> None:
    """Run the default HTML/JavaScript WebUI."""
    webui = HTMLWebUIBase(
        runtime_url,
        title=title,
        poll_interval=poll_interval,
    )
    webui.run(host=host, port=port)


def launch_html_webui_subprocess(
    runtime_url: str = "http://127.0.0.1:8010",
    *,
    host: str = "127.0.0.1",
    port: int = 8008,
    title: str = "EAA WebUI",
    poll_interval: float = 1.0,
    python_executable: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
) -> subprocess.Popen:
    """Launch the default HTML/JavaScript WebUI in a non-blocking subprocess."""
    executable = python_executable or sys.executable
    command = [
        executable,
        "-m",
        "eaa_core.gui.html",
        "--runtime-url",
        runtime_url,
        "--host",
        host,
        "--port",
        str(port),
        "--title",
        title,
        "--poll-interval",
        str(poll_interval),
    ]
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the WebUI module."""
    parser = argparse.ArgumentParser(prog="python -m eaa_core.gui.html")
    parser.add_argument(
        "--runtime-url",
        default="http://127.0.0.1:8010",
        help="Agent-side WebUI runtime URL.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=8008, type=int, help="Bind port.")
    parser.add_argument("--title", default="EAA WebUI", help="Browser and header title.")
    parser.add_argument(
        "--poll-interval",
        default=1.0,
        type=float,
        help="Message and status polling interval in seconds.",
    )
    return parser


def main() -> None:
    """Run the WebUI from command-line arguments."""
    args = build_parser().parse_args()
    run_html_webui(
        args.runtime_url,
        host=args.host,
        port=args.port,
        title=args.title,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
