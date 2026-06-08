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
    mathjax_route = "/static/mathjax"
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

    def page_html(self) -> str:
        """Return the complete HTML document for the browser UI."""
        config = {
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
                "mathjax": self.mathjax_route,
            },
        }
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(self.title)}</title>
  {self.styles()}
  <script>
    window.EAA_WEBUI_CONFIG = {json.dumps(config)};
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="{self.mathjax_route}/es5/tex-svg-full.js"></script>
</head>
<body>
  <div class="eaa-page">
    <header class="eaa-header">
      <div class="eaa-title">{html.escape(self.title)}</div>
      <div class="eaa-header-actions">
        <button id="interrupt-button" class="eaa-interrupt" type="button">Interrupt</button>
        <div id="connection-status" class="eaa-status">Connecting...</div>
      </div>
    </header>
    <main class="eaa-main">
      <section class="eaa-chat" aria-label="Chat transcript">
        <div id="messages" class="eaa-messages"></div>
      </section>
      <aside class="eaa-sidebar" aria-label="Images">
        <div class="eaa-sidebar-title">Images</div>
        <div id="images" class="eaa-images"></div>
      </aside>
    </main>
    <form id="input-form" class="eaa-input-panel">
      <div id="processing-status" class="eaa-processing hidden"></div>
      <div class="eaa-input-row">
        <textarea
          id="message-input"
          class="eaa-input"
          rows="3"
          placeholder="Type a message. Paste an image or use <img /absolute/path/to/image.png>."
        ></textarea>
        <div id="skill-suggestions" class="eaa-skill-suggestions hidden"></div>
        <button id="send-button" class="eaa-send" type="submit">Send</button>
      </div>
    </form>
  </div>
  <div id="image-preview" class="eaa-browser-image-preview" aria-hidden="true">
    <button class="eaa-browser-image-preview-close" type="button" aria-label="Close image preview">&times;</button>
    <img class="eaa-browser-image-preview-image" alt="">
  </div>
  <script>{self.script()}</script>
</body>
</html>"""

    def script(self) -> str:
        """Return browser JavaScript for the WebUI."""
        return r"""
(() => {
  const config = window.EAA_WEBUI_CONFIG;
  const routes = config.routes;
  const state = {
    lastMessageId: null,
    renderedIds: new Set(),
    pendingMessages: new Map(),
    pendingCounter: 0,
    skills: [],
    skillsLoaded: false,
    processing: false,
    eventSource: null
  };

  const messagesEl = document.getElementById('messages');
  const imagesEl = document.getElementById('images');
  const inputEl = document.getElementById('message-input');
  const formEl = document.getElementById('input-form');
  const sendButton = document.getElementById('send-button');
  const interruptButton = document.getElementById('interrupt-button');
  const connectionStatus = document.getElementById('connection-status');
  const processingStatus = document.getElementById('processing-status');
  const skillPanel = document.getElementById('skill-suggestions');

  const escapeHtml = (value) => String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const imageSource = (image) => {
    const value = String(image || '');
    if (value.startsWith('data:image')) return value;
    return `${routes.image}?path=${encodeURIComponent(value)}`;
  };

  const parseContentImagePaths = (content) => {
    const paths = [];
    const pattern = /<img\s+([^>\s]+)>/g;
    let match;
    while ((match = pattern.exec(String(content || ''))) !== null) {
      paths.push(match[1]);
    }
    return paths;
  };

  const inlineMarkdown = (text) => {
    let output = escapeHtml(text);
    const code = [];
    output = output.replace(/`([^`]+)`/g, (_match, value) => {
      const id = code.length;
      code.push(`<code>${value}</code>`);
      return `\u0000CODE${id}\u0000`;
    });
    output = output.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    output = output.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    output = output.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    output = output.replace(/\u0000CODE(\d+)\u0000/g, (_match, id) => code[Number(id)]);
    return output;
  };

  const renderTable = (lines) => {
    const rows = lines.map((line) => line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => inlineMarkdown(cell.trim())));
    const header = rows[0] || [];
    const body = rows.slice(2);
    return `<table><thead><tr>${header.map((cell) => `<th>${cell}</th>`).join('')}</tr></thead><tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  };

  const renderMarkdown = (markdown) => {
    const lines = String(markdown || '').replace(/\r\n/g, '\n').split('\n');
    const blocks = [];
    let paragraph = [];
    let code = null;
    let table = [];
    let list = null;

    const flushParagraph = () => {
      if (!paragraph.length) return;
      blocks.push(`<p>${inlineMarkdown(paragraph.join('\n')).replace(/\n/g, '<br>')}</p>`);
      paragraph = [];
    };
    const flushTable = () => {
      if (!table.length) return;
      blocks.push(renderTable(table));
      table = [];
    };
    const flushList = () => {
      if (!list) return;
      const tag = list.ordered ? 'ol' : 'ul';
      blocks.push(`<${tag}>${list.items.map((item) => `<li>${inlineMarkdown(item)}</li>`).join('')}</${tag}>`);
      list = null;
    };

    for (const line of lines) {
      const fence = line.match(/^```|^~~~/);
      if (fence && code === null) {
        flushParagraph();
        flushTable();
        flushList();
        code = [];
        continue;
      }
      if (fence && code !== null) {
        blocks.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
        code = null;
        continue;
      }
      if (code !== null) {
        code.push(line);
        continue;
      }
      const unorderedItem = line.match(/^\s*[-*+]\s+(.+)$/);
      const orderedItem = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (unorderedItem || orderedItem) {
        flushParagraph();
        flushTable();
        const ordered = Boolean(orderedItem);
        if (!list || list.ordered !== ordered) flushList();
        if (!list) list = {ordered, items: []};
        list.items.push((unorderedItem || orderedItem)[1]);
        continue;
      }
      if (/^\s*\|.+\|\s*$/.test(line)) {
        flushParagraph();
        flushList();
        table.push(line);
        continue;
      }
      flushTable();
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        flushList();
        const level = heading[1].length;
        blocks.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }
      if (/^\s*$/.test(line)) {
        flushParagraph();
        flushList();
        continue;
      }
      paragraph.push(line);
    }
    if (code !== null) blocks.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
    flushTable();
    flushList();
    flushParagraph();
    return blocks.join('\n');
  };

  const roleLabel = (role) => role === 'user_webui' ? 'user' : role;

  const isApprovalMessage = (message) => {
    return String(message.role || '') === 'system' && /Approve\?\s*\[y\/N\]:/i.test(String(message.content || ''));
  };

  const formatApproval = (content) => {
    const argsMatch = String(content).match(/Arguments:\s*([\s\S]*?)\nApprove\?\s*\[y\/N\]:/i);
    if (!argsMatch) return null;
    let args;
    try {
      args = JSON.parse(argsMatch[1]);
    } catch (_error) {
      return null;
    }
    const extracted = [];
    const scrub = (value, path = '') => {
      if (Array.isArray(value)) return value.map((item, index) => scrub(item, path ? `${path}[${index}]` : `[${index}]`));
      if (!value || typeof value !== 'object') return value;
      const next = {};
      for (const [key, item] of Object.entries(value)) {
        const childPath = path ? `${path}.${key}` : key;
        if ((key.toLowerCase() === 'code' || key.toLowerCase() === 'content') && typeof item === 'string') {
          extracted.push({label: childPath, value: item});
          next[key] = `<${childPath} rendered below>`;
        } else {
          next[key] = scrub(item, childPath);
        }
      }
      return next;
    };
    const toolMatch = String(content).match(/Tool\s+'([^']+)'\s+requires approval/i);
    return {
      summary: `Tool \`${toolMatch ? toolMatch[1] : 'tool'}\` requires approval before execution.`,
      args: JSON.stringify(scrub(args), null, 2),
      extracted
    };
  };

  const appendMarkdown = (container, content, role) => {
    const markdown = document.createElement('div');
    markdown.className = 'eaa-markdown';
    markdown.innerHTML = renderMarkdown(content);
    const lines = String(content || '').replace(/\r\n/g, '\n').split('\n');
    if (['user', 'user_webui', 'tool'].includes(role) && lines.length > 10) {
      markdown.classList.add('eaa-markdown-folded');
      const button = document.createElement('button');
      button.className = 'eaa-message-expand';
      button.type = 'button';
      button.textContent = 'Show more';
      button.addEventListener('click', () => {
        markdown.classList.remove('eaa-markdown-folded');
        button.remove();
      });
      container.append(markdown, button);
      return;
    }
    container.append(markdown);
  };

  const appendApproval = (container, content) => {
    const approval = formatApproval(content);
    if (!approval) {
      appendMarkdown(container, content, 'system');
      return;
    }
    appendMarkdown(container, approval.summary, 'system');
    const args = document.createElement('div');
    args.className = 'eaa-approval-arguments';
    args.innerHTML = `<div class="eaa-approval-section-title">Arguments</div><pre>${escapeHtml(approval.args)}</pre>`;
    container.append(args);
    for (const field of approval.extracted) {
      const extracted = document.createElement('div');
      extracted.className = 'eaa-approval-extracted-field';
      extracted.innerHTML = `<div class="eaa-approval-section-title">${escapeHtml(field.label)}</div><pre><code>${escapeHtml(field.value)}</code></pre>`;
      container.append(extracted);
    }
  };

  const appendToolCalls = (container, message) => {
    const toolCalls = message.tool_calls;
    if (toolCalls === null || toolCalls === undefined || String(toolCalls).trim() === '') return;
    const details = document.createElement('details');
    details.className = 'eaa-tool-call-details';
    details.open = true;
    details.innerHTML = `<summary>Tool calls</summary><pre>${escapeHtml(String(toolCalls).trim())}</pre>`;
    container.append(details);
  };

  const appendApprovalActions = (container, message) => {
    if (!isApprovalMessage(message)) return;
    const row = document.createElement('div');
    row.className = 'eaa-approval-actions';
    const submit = async (value) => {
      for (const button of row.querySelectorAll('button')) button.disabled = true;
      await fetch(routes.approval, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({approved: value === 'yes'})
      });
    };
    for (const [label, value, cls] of [['Yes', 'yes', 'eaa-approval-yes'], ['No', 'no', 'eaa-approval-no']]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `eaa-approval-button ${cls}`;
      button.textContent = label;
      button.addEventListener('click', () => submit(value));
      row.append(button);
    }
    container.append(row);
  };

  const appendImages = (messageEl, message) => {
    const sources = [];
    const seen = new Set();
    const attached = Array.isArray(message.images) && message.images.length ? message.images : (message.image ? [message.image] : []);
    for (const image of attached) {
      const source = imageSource(image);
      if (!seen.has(source)) {
        seen.add(source);
        sources.push(source);
      }
    }
    if (String(message.role || '') !== 'system') {
      for (const path of parseContentImagePaths(message.content)) {
        const source = imageSource(path);
        if (!seen.has(source)) {
          seen.add(source);
          sources.push(source);
        }
      }
    }
    if (!sources.length) return false;
    const row = document.createElement('div');
    row.className = 'eaa-message-images';
    for (const source of sources) {
      row.append(createImage(source, 'eaa-message-image eaa-clickable-image'));
      imagesEl.append(createImage(source, 'eaa-sidebar-image eaa-clickable-image'));
    }
    messageEl.append(row);
    followScroll(imagesEl);
    return true;
  };

  const createImage = (source, className) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'eaa-image-html';
    const image = document.createElement('img');
    image.src = source;
    image.className = className;
    image.dataset.eaaFullSrc = source;
    image.loading = 'lazy';
    image.decoding = 'async';
    image.alt = '';
    wrapper.append(image);
    return wrapper;
  };

  const renderMessage = (message) => {
    const role = String(message.role || 'message');
    const content = String(message.content || '').trim();
    const messageEl = document.createElement('article');
    messageEl.className = `eaa-message eaa-message-${role}`;
    const roleEl = document.createElement('div');
    roleEl.className = 'eaa-role';
    roleEl.textContent = roleLabel(role);
    messageEl.append(roleEl);
    if (content) {
      if (isApprovalMessage(message)) appendApproval(messageEl, content);
      else appendMarkdown(messageEl, content, role);
    }
    appendToolCalls(messageEl, message);
    appendImages(messageEl, message);
    appendApprovalActions(messageEl, message);
    messagesEl.append(messageEl);
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([messageEl]).catch((error) => console.warn('MathJax typesetting failed:', error));
    }
  };

  const followScroll = (el) => {
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const shouldFollow = !el.dataset.eaaUserScrolled || distance <= 100;
    if (shouldFollow) requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  };

  for (const el of [messagesEl, imagesEl]) {
    el.addEventListener('scroll', () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      el.dataset.eaaUserScrolled = distance > 100 ? '1' : '';
    }, {passive: true});
  }

  const consumePending = (message) => {
    const role = message.role;
    if (!['user', 'user_webui'].includes(role)) return false;
    const content = String(message.content || '').trim();
    for (const [id, pendingContent] of state.pendingMessages.entries()) {
      if (pendingContent === content) {
        state.pendingMessages.delete(id);
        return true;
      }
    }
    return false;
  };

  const renderMessages = (messages) => {
    let rendered = false;
    for (const message of messages) {
      const messageId = String(message.id ?? '');
      if (state.renderedIds.has(messageId)) continue;
      if (consumePending(message)) {
        state.renderedIds.add(messageId);
        if (Number.isInteger(message.id)) state.lastMessageId = message.id;
        continue;
      }
      renderMessage(message);
      state.renderedIds.add(messageId);
      if (Number.isInteger(message.id)) state.lastMessageId = message.id;
      rendered = true;
    }
    if (rendered) followScroll(messagesEl);
  };

  const applyStatus = (payload) => {
    updateProcessingStatus(Boolean(payload.input_requested), payload.status || 'idle');
    if (payload.interrupt_requested) connectionStatus.textContent = 'Interrupt requested';
  };

  const loadState = async () => {
    try {
      const response = await fetch(routes.state);
      if (!response.ok) throw new Error('State fetch failed');
      const payload = await response.json();
      renderMessages(Array.isArray(payload.messages) ? payload.messages : []);
      applyStatus(payload);
      connectionStatus.textContent = 'Connected';
    } catch (_error) {
      connectionStatus.textContent = 'Reconnecting...';
    }
  };

  const connectEvents = () => {
    if (state.eventSource) state.eventSource.close();
    state.eventSource = new EventSource(routes.events);
    state.eventSource.onopen = () => {
      connectionStatus.textContent = 'Connected';
    };
    state.eventSource.onerror = () => {
      connectionStatus.textContent = 'Reconnecting...';
    };
    state.eventSource.addEventListener('message.created', (event) => {
      const payload = JSON.parse(event.data || '{}');
      if (payload.message) renderMessages([payload.message]);
    });
    state.eventSource.addEventListener('status.changed', (event) => {
      applyStatus(JSON.parse(event.data || '{}'));
    });
    state.eventSource.addEventListener('interrupt.requested', (event) => {
      applyStatus(JSON.parse(event.data || '{}'));
    });
    state.eventSource.addEventListener('interrupt.cleared', (event) => {
      applyStatus(JSON.parse(event.data || '{}'));
    });
    state.eventSource.addEventListener('input.requested', () => {
      updateProcessingStatus(true, 'waiting_for_input');
    });
    state.eventSource.addEventListener('approval.requested', (event) => {
      const payload = JSON.parse(event.data || '{}');
      const content = `Tool '${payload.tool_name || 'tool'}' requires approval before execution.\nArguments: ${JSON.stringify(payload.arguments || {}, null, 2)}\nApprove? [y/N]: `;
      renderMessage({id: `approval-${Date.now()}`, role: 'system', content});
    });
  };

  const updateProcessingStatus = (inputRequested, statusText = 'idle') => {
    state.processing = !inputRequested && statusText !== 'idle';
    sendButton.disabled = state.processing;
    interruptButton.disabled = statusText === 'idle';
    if (state.processing) {
      processingStatus.textContent = 'Agent is processing...';
      processingStatus.classList.remove('hidden');
    } else {
      processingStatus.textContent = '';
      processingStatus.classList.add('hidden');
    }
  };

  interruptButton.addEventListener('click', async () => {
    interruptButton.disabled = true;
    await fetch(routes.interrupt, {method: 'POST'});
  });

  const queueMessage = async (content) => {
    const response = await fetch(routes.send, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({content})
    });
    if (!response.ok) throw new Error('Send failed');
  };

  const appendPendingMessage = (content) => {
    const id = `pending-${state.pendingCounter++}`;
    state.pendingMessages.set(id, String(content).trim());
    renderMessage({id, role: 'user', content, images: [], pending: true});
    followScroll(messagesEl);
  };

  const sendCurrentMessage = async () => {
    if (state.processing) return;
    const content = inputEl.value.trim();
    if (!content) return;
    sendButton.disabled = true;
    try {
      await queueMessage(content);
      appendPendingMessage(content);
      inputEl.value = '';
      hideSkillSuggestions();
    } finally {
      sendButton.disabled = state.processing;
    }
  };

  formEl.addEventListener('submit', (event) => {
    event.preventDefault();
    sendCurrentMessage();
  });

  inputEl.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    sendCurrentMessage();
  });

  document.addEventListener('paste', async (event) => {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (!item.type || !item.type.startsWith('image/')) continue;
      const file = item.getAsFile();
      if (!file) continue;
      event.preventDefault();
      const reader = new FileReader();
      reader.onload = async () => {
        const response = await fetch(routes.upload, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({image_data: reader.result})
        });
        if (!response.ok) return;
        const result = await response.json();
        const tag = `<img ${result.file_path}> `;
        const start = inputEl.selectionStart || inputEl.value.length;
        const end = inputEl.selectionEnd || inputEl.value.length;
        inputEl.value = inputEl.value.slice(0, start) + tag + inputEl.value.slice(end);
        inputEl.selectionStart = inputEl.selectionEnd = start + tag.length;
        inputEl.focus();
      };
      reader.readAsDataURL(file);
      break;
    }
  });

  const closePreview = () => {
    const preview = document.getElementById('image-preview');
    const image = preview.querySelector('img');
    preview.classList.remove('open');
    preview.setAttribute('aria-hidden', 'true');
    image.removeAttribute('src');
  };

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.closest('.eaa-browser-image-preview-close') || target.id === 'image-preview') {
      closePreview();
      return;
    }
    const image = target.closest('img[data-eaa-full-src]');
    if (!image) return;
    const preview = document.getElementById('image-preview');
    const previewImage = preview.querySelector('img');
    previewImage.src = image.dataset.eaaFullSrc || image.src;
    preview.classList.add('open');
    preview.setAttribute('aria-hidden', 'false');
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closePreview();
  });

  const refreshSkills = async () => {
    if (state.skillsLoaded) return;
    const response = await fetch(routes.skillCatalog);
    if (!response.ok) return;
    const payload = await response.json();
    state.skills = Array.isArray(payload.skills) ? payload.skills : [];
    state.skillsLoaded = true;
  };

  const hideSkillSuggestions = () => {
    skillPanel.classList.add('hidden');
    skillPanel.innerHTML = '';
  };

  const renderSkillSuggestions = async () => {
    const value = inputEl.value || '';
    if (!value.startsWith('/skill')) {
      hideSkillSuggestions();
      return;
    }
    await refreshSkills();
    const typed = value.slice('/skill'.length).trimStart().toLowerCase();
    const matches = state.skills
      .filter((skill) => !typed || String(skill.name || '').toLowerCase().startsWith(typed))
      .slice(0, 8);
    if (!matches.length) {
      hideSkillSuggestions();
      return;
    }
    skillPanel.innerHTML = matches.map((skill) => `
      <button type="button" class="eaa-skill-suggestion" data-skill="${escapeHtml(skill.name || '')}">
        <span class="eaa-skill-name">${escapeHtml(skill.name || '')}</span>
        <span class="eaa-skill-description">${escapeHtml(skill.description || '')}</span>
      </button>
    `).join('');
    skillPanel.classList.remove('hidden');
  };

  inputEl.addEventListener('input', renderSkillSuggestions);
  inputEl.addEventListener('blur', () => setTimeout(hideSkillSuggestions, 150));
  skillPanel.addEventListener('mousedown', (event) => {
    const button = event.target.closest('.eaa-skill-suggestion');
    if (!button) return;
    event.preventDefault();
    inputEl.value = `/skill ${button.dataset.skill} `;
    inputEl.focus();
    hideSkillSuggestions();
  });

  loadState();
  connectEvents();
})();
"""

    def styles(self) -> str:
        """Return CSS for the WebUI."""
        return """
<style>
html, body { height: 100%; margin: 0; }
body {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7f9;
  color: #20242a;
}
button, textarea { font: inherit; }
.eaa-page { height: 100vh; width: 100%; display: flex; flex-direction: column; }
.eaa-header {
  min-height: 52px; display: flex; align-items: center; justify-content: space-between;
  padding: 0 18px; border-bottom: 1px solid #d9dde3; background: #ffffff;
}
.eaa-title { font-size: 18px; font-weight: 650; }
.eaa-header-actions { display: flex; align-items: center; gap: 10px; }
.eaa-status { color: #586170; font-size: 13px; }
.eaa-interrupt {
  padding: 6px 10px; border: 1px solid #c2410c; border-radius: 6px;
  background: #fff7ed; color: #9a3412; cursor: pointer;
}
.eaa-interrupt:disabled { border-color: #e5e7eb; background: #f3f4f6; color: #8a94a3; cursor: not-allowed; }
.eaa-main { flex: 1; min-height: 0; display: flex; }
.eaa-chat { flex: 1; min-width: 0; height: 100%; border-right: 1px solid #d9dde3; }
.eaa-messages { height: 100%; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; padding: 16px; box-sizing: border-box; }
.eaa-sidebar { width: 280px; min-width: 220px; height: 100%; background: #ffffff; padding: 12px; box-sizing: border-box; display: flex; flex-direction: column; gap: 10px; }
.eaa-sidebar-title { font-size: 13px; font-weight: 650; color: #586170; }
.eaa-images { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
.eaa-message {
  max-width: 860px; min-width: 0; display: flex; flex-direction: column; gap: 6px;
  padding: 10px 12px; border: 1px solid #dfe3e8; border-radius: 8px; background: #ffffff;
}
.eaa-message-user, .eaa-message-user_webui { align-self: flex-end; background: #eef6ff; border-color: #c9ddf2; }
.eaa-message-assistant { background: #f7f9fc; border-color: #e1e6ee; }
.eaa-message-system { background: #fff7e6; border-color: #f0d9a8; }
.eaa-message-tool { background: #f1f6f0; border-color: #d3e2cf; }
.eaa-role { font-size: 12px; color: #667085; font-weight: 650; text-transform: uppercase; }
.eaa-markdown { font-size: 14px; line-height: 1.45; min-width: 0; max-width: 100%; overflow-wrap: anywhere; word-break: break-word; }
.eaa-markdown p { margin: 0 0 0.65rem; }
.eaa-markdown p:last-child { margin-bottom: 0; }
.eaa-markdown ul, .eaa-markdown ol { margin: 0.2rem 0 0.65rem 1.25rem; padding: 0; }
.eaa-markdown li { margin: 0.15rem 0; }
.eaa-markdown h1, .eaa-markdown h2, .eaa-markdown h3, .eaa-markdown h4, .eaa-markdown h5, .eaa-markdown h6 { margin: 0.25rem 0 0.5rem; line-height: 1.25; }
.eaa-markdown pre {
  margin: 0.75rem 0; padding: 0.875rem 1rem; overflow-x: auto;
  background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 8px;
}
.eaa-markdown pre code { background: transparent; padding: 0; }
.eaa-markdown code:not(pre code) { background: #f3f4f6; padding: 0.1rem 0.3rem; border-radius: 4px; }
.eaa-markdown table { border-collapse: collapse; max-width: 100%; overflow-x: auto; display: block; }
.eaa-markdown th, .eaa-markdown td { border: 1px solid #d7dde8; padding: 5px 7px; text-align: left; }
.eaa-message-images { display: flex; gap: 8px; flex-wrap: wrap; }
.eaa-message-image {
  display: block; width: 234px; max-height: 234px; object-fit: contain;
  border: 1px solid #d9dde3; border-radius: 6px; background: #ffffff;
}
.eaa-sidebar-image {
  display: block; width: 100%; max-height: 220px; object-fit: contain; flex-shrink: 0;
  border: 1px solid #d9dde3; border-radius: 6px; background: #ffffff;
}
.eaa-image-html { line-height: 0; }
.eaa-clickable-image { cursor: zoom-in; }
.eaa-markdown-folded { max-height: 16rem; overflow: hidden; }
.eaa-message-expand {
  align-self: flex-start; border: 0; background: transparent; color: #2563eb;
  padding: 2px 0; cursor: pointer;
}
.eaa-tool-call-details { width: 100%; font-size: 13px; }
.eaa-tool-call-details summary { color: #475467; cursor: pointer; font-weight: 650; user-select: none; }
.eaa-tool-call-details pre {
  max-height: 20rem; overflow: auto; margin: 6px 0 0; padding: 8px 10px;
  white-space: pre-wrap; overflow-wrap: anywhere; background: #eef2f7;
  border: 1px solid #d7dde8; border-radius: 6px;
}
.eaa-approval-actions { display: flex; gap: 8px; margin-top: 4px; }
.eaa-approval-button { min-width: 72px; padding: 7px 12px; border-radius: 6px; border: 1px solid #cbd5e1; background: #ffffff; cursor: pointer; }
.eaa-approval-button:disabled { color: #8a94a3; background: #e5e7eb; cursor: not-allowed; }
.eaa-approval-arguments, .eaa-approval-extracted-field { width: 100%; }
.eaa-approval-section-title { margin: 2px 0 4px; color: #475467; font-size: 12px; font-weight: 650; text-transform: uppercase; }
.eaa-approval-arguments pre, .eaa-approval-extracted-field pre {
  max-height: 24rem; overflow: auto; margin: 0; padding: 8px 10px; white-space: pre-wrap;
  overflow-wrap: anywhere; background: #f8fafc; border: 1px solid #d7dde8; border-radius: 6px;
}
.eaa-approval-extracted-field pre { background: #111827; color: #f9fafb; }
.eaa-browser-image-preview {
  position: fixed; inset: 0; z-index: 10000; display: none; align-items: center;
  justify-content: center; padding: 24px; background: rgba(15, 23, 42, 0.72);
}
.eaa-browser-image-preview.open { display: flex; }
.eaa-browser-image-preview-image {
  display: block; max-width: 92vw; max-height: 92vh; object-fit: contain;
  background: #ffffff; border-radius: 6px;
}
.eaa-browser-image-preview-close {
  position: absolute; top: 16px; right: 18px; width: 36px; height: 36px;
  border: 0; border-radius: 18px; background: #ffffff; color: #20242a;
  font-size: 24px; line-height: 36px; cursor: pointer;
}
.eaa-input-panel {
  margin: 12px 16px 16px; padding: 12px 14px 14px; border: 1px solid #d9dde3;
  border-radius: 10px; background: #ffffff; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.eaa-processing { color: #8a4b00; font-size: 13px; margin-bottom: 8px; }
.hidden { display: none; }
.eaa-input-row { display: flex; align-items: stretch; gap: 8px; position: relative; }
.eaa-input {
  flex: 1; min-width: 0; resize: vertical; max-height: 220px; padding: 9px 10px;
  border: 1px solid #cfd6df; border-radius: 8px; box-sizing: border-box;
}
.eaa-skill-suggestions {
  position: absolute; left: 0; right: 104px; bottom: 100%; margin-bottom: 6px; z-index: 20;
  max-height: 260px; overflow-y: auto; background: #ffffff; border: 1px solid #d8dde6;
  border-radius: 8px; box-shadow: 0 10px 24px rgba(21, 30, 45, 0.14); padding: 6px;
}
.eaa-skill-suggestion { display: block; width: 100%; border: 0; background: transparent; text-align: left; padding: 8px 10px; border-radius: 6px; cursor: pointer; }
.eaa-skill-suggestion:hover { background: #eef3fb; }
.eaa-skill-name { display: block; font-weight: 650; color: #20242a; }
.eaa-skill-description { display: block; margin-top: 2px; color: #5f6b7a; font-size: 12px; line-height: 1.35; }
.eaa-send {
  min-width: 96px; padding: 0 16px; border: 1px solid #1f5fbf; border-radius: 8px;
  background: #2563eb; color: #ffffff; cursor: pointer;
}
.eaa-send:disabled { border-color: #e5e7eb; background: #e5e7eb; color: #8a94a3; cursor: not-allowed; }
@media (max-width: 800px) {
  .eaa-main { flex-direction: column; }
  .eaa-sidebar { width: 100%; height: 190px; border-top: 1px solid #d9dde3; }
  .eaa-chat { border-right: none; }
}
</style>
"""


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
