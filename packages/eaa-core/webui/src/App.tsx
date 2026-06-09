import { FormEvent, KeyboardEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleStop, Send, X } from "lucide-react";

import { escapeHtml, renderMarkdown } from "./markdown";
import type { PendingApproval, RuntimeSnapshot, Skill, WebUIConfig, WebUIMessage } from "./types";
import "./styles.css";

type ConnectionState = "Connecting..." | "Connected" | "Reconnecting..." | "Interrupt requested";

const defaultConfig: WebUIConfig = {
  title: "EAA WebUI",
  runtimeUrl: "http://127.0.0.1:8010",
  pollIntervalMs: 1000,
  routes: {
    events: "/api/events",
    state: "/api/state",
    image: "/api/image",
    send: "/api/input",
    interrupt: "/api/interrupt",
    approval: "/api/approval",
    upload: "/api/upload-image",
    skillCatalog: "/api/skill-catalog",
    mathjax: "/static/mathjax",
  },
};

const config = window.EAA_WEBUI_CONFIG ?? defaultConfig;

const roleLabel = (role: string) => (role === "user_webui" ? "user" : role);

const parseContentImagePaths = (content: unknown): string[] => {
  const paths: string[] = [];
  const pattern = /<img\s+([^>\s]+)>/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(String(content ?? ""))) !== null) paths.push(match[1]);
  return paths;
};

const imageSource = (image: unknown) => {
  const value = String(image ?? "");
  if (value.startsWith("data:image")) return value;
  return `${config.routes.image}?path=${encodeURIComponent(value)}`;
};

const isApprovalMessage = (message: WebUIMessage) =>
  String(message.role ?? "") === "system" && /Approve\?\s*\[y\/N\]:/i.test(String(message.content ?? ""));

const formatApproval = (content: unknown) => {
  const text = String(content ?? "");
  const argsMatch = text.match(/Arguments:\s*([\s\S]*?)\nApprove\?\s*\[y\/N\]:/i);
  if (!argsMatch) return null;
  let args: unknown;
  try {
    args = JSON.parse(argsMatch[1]);
  } catch (_error) {
    return null;
  }
  const extracted: { label: string; value: string }[] = [];
  const scrub = (value: unknown, path = ""): unknown => {
    if (Array.isArray(value)) return value.map((item, index) => scrub(item, path ? `${path}[${index}]` : `[${index}]`));
    if (!value || typeof value !== "object") return value;
    const next: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      const childPath = path ? `${path}.${key}` : key;
      if ((key.toLowerCase() === "code" || key.toLowerCase() === "content") && typeof item === "string") {
        extracted.push({ label: childPath, value: item });
        next[key] = `<${childPath} rendered below>`;
      } else {
        next[key] = scrub(item, childPath);
      }
    }
    return next;
  };
  const toolMatch = text.match(/Tool\s+'([^']+)'\s+requires approval/i);
  return {
    summary: `Tool \`${toolMatch ? toolMatch[1] : "tool"}\` requires approval before execution.`,
    args: JSON.stringify(scrub(args), null, 2),
    extracted,
  };
};

const messageKey = (message: WebUIMessage, index: number) => String(message.id ?? `message-${index}`);

function MarkdownBlock({ content, role }: { content: unknown; role: string }) {
  const [expanded, setExpanded] = useState(false);
  const lines = String(content ?? "").replace(/\r\n/g, "\n").split("\n");
  const folded = !expanded && ["user", "user_webui", "tool"].includes(role) && lines.length > 10;
  return (
    <>
      <div
        className={`eaa-markdown${folded ? " eaa-markdown-folded" : ""}`}
        dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
      />
      {folded ? (
        <button className="eaa-message-expand" type="button" onClick={() => setExpanded(true)}>
          Show more
        </button>
      ) : null}
    </>
  );
}

function ApprovalContent({ content }: { content: unknown }) {
  const approval = formatApproval(content);
  if (!approval) return <MarkdownBlock content={content} role="system" />;
  return (
    <>
      <MarkdownBlock content={approval.summary} role="system" />
      <div className="eaa-approval-arguments">
        <div className="eaa-approval-section-title">Arguments</div>
        <pre>{approval.args}</pre>
      </div>
      {approval.extracted.map((field) => (
        <div className="eaa-approval-extracted-field" key={field.label}>
          <div className="eaa-approval-section-title">{field.label}</div>
          <pre>
            <code>{field.value}</code>
          </pre>
        </div>
      ))}
    </>
  );
}

function CodeBlock({ content }: { content: unknown }) {
  return (
    <div className="eaa-markdown eaa-tool-response">
      <pre>
        <code>{String(content ?? "")}</code>
      </pre>
    </div>
  );
}

function MessageView({
  message,
  onImage,
  onApproval,
  registerImages,
}: {
  message: WebUIMessage;
  onImage: (src: string) => void;
  onApproval: (approved: boolean) => Promise<void>;
  registerImages: (sources: string[]) => void;
}) {
  const [approvalSubmitted, setApprovalSubmitted] = useState(false);
  const role = String(message.role ?? "message");
  const content = String(message.content ?? "").trim();
  const imageSources = useMemo(() => {
    const sources: string[] = [];
    const seen = new Set<string>();
    const attached = Array.isArray(message.images) && message.images.length ? message.images : message.image ? [message.image] : [];
    for (const image of attached) {
      const source = imageSource(image);
      if (!seen.has(source)) {
        seen.add(source);
        sources.push(source);
      }
    }
    if (role !== "system") {
      for (const path of parseContentImagePaths(message.content)) {
        const source = imageSource(path);
        if (!seen.has(source)) {
          seen.add(source);
          sources.push(source);
        }
      }
    }
    return sources;
  }, [message, role]);

  useEffect(() => {
    registerImages(imageSources);
  }, [imageSources, registerImages]);

  const submitApproval = async (approved: boolean) => {
    setApprovalSubmitted(true);
    await onApproval(approved);
  };

  return (
    <article className={`eaa-message eaa-message-${role}`}>
      <div className="eaa-role">{roleLabel(role)}</div>
      {content ? (
        isApprovalMessage(message) ? (
          <ApprovalContent content={content} />
        ) : role === "tool" ? (
          <CodeBlock content={content} />
        ) : (
          <MarkdownBlock content={content} role={role} />
        )
      ) : null}
      {message.tool_calls !== null && message.tool_calls !== undefined && String(message.tool_calls).trim() !== "" ? (
        <details className="eaa-tool-call-details" open>
          <summary>Tool calls</summary>
          <pre>{String(message.tool_calls).trim()}</pre>
        </details>
      ) : null}
      {imageSources.length ? (
        <div className="eaa-message-images">
          {imageSources.map((source) => (
            <button className="eaa-image-button" key={source} type="button" onClick={() => onImage(source)}>
              <img className="eaa-message-image" src={source} loading="lazy" decoding="async" alt="" />
            </button>
          ))}
        </div>
      ) : null}
      {isApprovalMessage(message) ? (
        <div className="eaa-approval-actions">
          <button
            className="eaa-approval-button eaa-approval-yes"
            disabled={approvalSubmitted}
            type="button"
            onClick={() => submitApproval(true)}
          >
            Yes
          </button>
          <button
            className="eaa-approval-button eaa-approval-no"
            disabled={approvalSubmitted}
            type="button"
            onClick={() => submitApproval(false)}
          >
            No
          </button>
        </div>
      ) : null}
    </article>
  );
}

function App() {
  const [messages, setMessages] = useState<WebUIMessage[]>([]);
  const [pendingCounter, setPendingCounter] = useState(0);
  const [connection, setConnection] = useState<ConnectionState>("Connecting...");
  const [status, setStatus] = useState("idle");
  const [inputRequested, setInputRequested] = useState(false);
  const [interruptRequested, setInterruptRequested] = useState(false);
  const [infoMessage, setInfoMessage] = useState<{ id: number; text: string } | null>(null);
  const [content, setContent] = useState("");
  const [skills, setSkills] = useState<Skill[]>([]);
  const [skillsLoaded, setSkillsLoaded] = useState(false);
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [sidebarImages, setSidebarImages] = useState<string[]>([]);
  const messagesRef = useRef<HTMLDivElement>(null);
  const imagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const renderedIdsRef = useRef<Set<string>>(new Set());
  const pendingMessagesRef = useRef<Map<string, string>>(new Map());
  const infoTimeoutRef = useRef<number | null>(null);

  const processing = !inputRequested && status !== "idle";

  const followScroll = useCallback((el: HTMLDivElement | null) => {
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (!el.dataset.eaaUserScrolled || distance <= 100) {
      requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
      });
    }
  }, []);

  const registerScrollIntent = (event: React.UIEvent<HTMLDivElement>) => {
    const el = event.currentTarget;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    el.dataset.eaaUserScrolled = distance > 100 ? "1" : "";
  };

  const registerImages = useCallback((sources: string[]) => {
    if (!sources.length) return;
    setSidebarImages((previous) => {
      const seen = new Set(previous);
      const next = [...previous];
      for (const source of sources) {
        if (!seen.has(source)) {
          seen.add(source);
          next.push(source);
        }
      }
      return next;
    });
  }, []);

  useEffect(() => {
    followScroll(messagesRef.current);
  }, [messages, followScroll]);

  useEffect(() => {
    followScroll(imagesRef.current);
  }, [sidebarImages, followScroll]);

  useEffect(() => {
    if (!window.MathJax?.typesetPromise || !messagesRef.current) return;
    window.MathJax.typesetPromise([messagesRef.current]).catch((error) => console.warn("MathJax typesetting failed:", error));
  }, [messages]);

  const consumePending = useCallback(
    (message: WebUIMessage) => {
      const role = message.role;
      if (!["user", "user_webui"].includes(String(role))) return false;
      const trimmed = String(message.content ?? "").trim();
      for (const [id, pendingContent] of pendingMessagesRef.current.entries()) {
        if (pendingContent === trimmed) {
          pendingMessagesRef.current.delete(id);
          return true;
        }
      }
      return false;
    },
    [],
  );

  const mergeMessages = useCallback(
    (incoming: WebUIMessage[]) => {
      const toRender: WebUIMessage[] = [];
      incoming.forEach((message, index) => {
        const id = messageKey(message, index);
        if (renderedIdsRef.current.has(id)) return;
        renderedIdsRef.current.add(id);
        if (!consumePending(message)) toRender.push(message);
      });
      if (toRender.length) setMessages((previous) => [...previous, ...toRender]);
    },
    [consumePending],
  );

  const applyStatus = useCallback((payload: RuntimeSnapshot) => {
    setStatus(payload.status ?? "idle");
    setInputRequested(Boolean(payload.input_requested));
    setInterruptRequested(Boolean(payload.interrupt_requested));
    if (payload.interrupt_requested) setConnection("Interrupt requested");
  }, []);

  const renderApprovalRequest = useCallback((payload: PendingApproval) => {
    const approvalContent = `Tool '${payload.tool_name || "tool"}' requires approval before execution.\nArguments: ${JSON.stringify(
      payload.arguments || {},
      null,
      2,
    )}\nApprove? [y/N]: `;
    mergeMessages([{ id: `approval-${Date.now()}`, role: "system", content: approvalContent }]);
  }, [mergeMessages]);

  const loadState = useCallback(async () => {
    try {
      const response = await fetch(config.routes.state);
      if (!response.ok) throw new Error("State fetch failed");
      const payload = (await response.json()) as RuntimeSnapshot;
      mergeMessages(Array.isArray(payload.messages) ? payload.messages : []);
      applyStatus(payload);
      if (payload.pending_approval) renderApprovalRequest(payload.pending_approval);
      setConnection("Connected");
    } catch (_error) {
      setConnection("Reconnecting...");
    }
  }, [applyStatus, mergeMessages, renderApprovalRequest]);

  useEffect(() => {
    loadState();
  }, [loadState]);

  useEffect(() => {
    const source = new EventSource(config.routes.events);
    source.onopen = () => setConnection("Connected");
    source.onerror = () => setConnection("Reconnecting...");
    source.addEventListener("message.created", (event) => {
      const payload = JSON.parse(event.data || "{}") as { message?: WebUIMessage };
      if (payload.message) mergeMessages([payload.message]);
    });
    source.addEventListener("status.changed", (event) => applyStatus(JSON.parse(event.data || "{}") as RuntimeSnapshot));
    source.addEventListener("interrupt.requested", (event) => applyStatus(JSON.parse(event.data || "{}") as RuntimeSnapshot));
    source.addEventListener("interrupt.cleared", (event) => applyStatus(JSON.parse(event.data || "{}") as RuntimeSnapshot));
    source.addEventListener("input.requested", () => {
      setStatus("waiting_for_input");
      setInputRequested(true);
    });
    source.addEventListener("approval.requested", (event) => renderApprovalRequest(JSON.parse(event.data || "{}") as PendingApproval));
    return () => source.close();
  }, [applyStatus, mergeMessages, renderApprovalRequest]);

  const submitApproval = async (approved: boolean) => {
    await fetch(config.routes.approval, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved }),
    });
  };

  const appendPendingMessage = (pendingContent: string) => {
    const id = `pending-${pendingCounter}`;
    setPendingCounter((value) => value + 1);
    pendingMessagesRef.current.set(id, pendingContent.trim());
    setMessages((previous) => [...previous, { id, role: "user", content: pendingContent, images: [], pending: true }]);
    renderedIdsRef.current.add(id);
  };

  const sendCurrentMessage = async () => {
    if (processing) return;
    const trimmed = content.trim();
    if (!trimmed) return;
    const response = await fetch(config.routes.send, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: trimmed }),
    });
    if (!response.ok) throw new Error("Send failed");
    appendPendingMessage(trimmed);
    setContent("");
    setSuggestionsOpen(false);
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void sendCurrentMessage();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void sendCurrentMessage();
  };

  const requestInterrupt = async () => {
    setInterruptRequested(true);
    if (infoTimeoutRef.current !== null) window.clearTimeout(infoTimeoutRef.current);
    setInfoMessage({
      id: Date.now(),
      text: "Workflow will be interrupted after the next LLM or tool response.",
    });
    infoTimeoutRef.current = window.setTimeout(() => {
      setInfoMessage(null);
      infoTimeoutRef.current = null;
    }, 5500);
    await fetch(config.routes.interrupt, { method: "POST" });
  };

  const refreshSkills = async () => {
    if (skillsLoaded) return;
    const response = await fetch(config.routes.skillCatalog);
    if (!response.ok) return;
    const payload = (await response.json()) as { skills?: Skill[] };
    setSkills(Array.isArray(payload.skills) ? payload.skills : []);
    setSkillsLoaded(true);
  };

  const onInputChange = async (value: string) => {
    setContent(value);
    if (!value.startsWith("/skill")) {
      setSuggestionsOpen(false);
      return;
    }
    await refreshSkills();
    setSuggestionsOpen(true);
  };

  const skillMatches = useMemo(() => {
    if (!content.startsWith("/skill")) return [];
    const typed = content.slice("/skill".length).trimStart().toLowerCase();
    return skills.filter((skill) => !typed || String(skill.name ?? "").toLowerCase().startsWith(typed)).slice(0, 8);
  }, [content, skills]);

  const chooseSkill = (event: MouseEvent<HTMLButtonElement>, skill: Skill) => {
    event.preventDefault();
    setContent(`/skill ${skill.name ?? ""} `);
    setSuggestionsOpen(false);
    requestAnimationFrame(() => inputRef.current?.focus());
  };

  const onPaste = (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = event.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (!item.type || !item.type.startsWith("image/")) continue;
      const file = item.getAsFile();
      if (!file) continue;
      event.preventDefault();
      const reader = new FileReader();
      reader.onload = async () => {
        const response = await fetch(config.routes.upload, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_data: reader.result }),
        });
        if (!response.ok) return;
        const result = (await response.json()) as { file_path?: string };
        const tag = `<img ${result.file_path}> `;
        const textarea = inputRef.current;
        const start = textarea?.selectionStart ?? content.length;
        const end = textarea?.selectionEnd ?? content.length;
        setContent((value) => value.slice(0, start) + tag + value.slice(end));
        requestAnimationFrame(() => {
          if (!textarea) return;
          textarea.selectionStart = textarea.selectionEnd = start + tag.length;
          textarea.focus();
        });
      };
      reader.readAsDataURL(file);
      break;
    }
  };

  useEffect(() => {
    const close = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setPreviewImage(null);
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, []);

  useEffect(() => {
    return () => {
      if (infoTimeoutRef.current !== null) window.clearTimeout(infoTimeoutRef.current);
    };
  }, []);

  return (
    <div className="eaa-page">
      {infoMessage ? (
        <div className="eaa-info-box" key={infoMessage.id}>
          {infoMessage.text}
        </div>
      ) : null}
      <header className="eaa-header">
        <div className="eaa-title">{config.title}</div>
        <div className="eaa-header-actions">
          <button className="eaa-interrupt" type="button" disabled={status === "idle" && !interruptRequested} onClick={() => void requestInterrupt()}>
            <CircleStop size={16} aria-hidden="true" />
            <span>Interrupt</span>
          </button>
          <div className="eaa-status">{connection}</div>
        </div>
      </header>
      <main className="eaa-main">
        <section className="eaa-chat" aria-label="Chat transcript">
          <div className="eaa-messages" ref={messagesRef} onScroll={registerScrollIntent}>
            {messages.map((message, index) => (
              <MessageView
                key={messageKey(message, index)}
                message={message}
                onImage={setPreviewImage}
                onApproval={submitApproval}
                registerImages={registerImages}
              />
            ))}
          </div>
        </section>
        <aside className="eaa-sidebar" aria-label="Images">
          <div className="eaa-sidebar-title">Images</div>
          <div className="eaa-images" ref={imagesRef} onScroll={registerScrollIntent}>
            {sidebarImages.map((source) => (
              <button className="eaa-image-button" key={source} type="button" onClick={() => setPreviewImage(source)}>
                <img className="eaa-sidebar-image" src={source} loading="lazy" decoding="async" alt="" />
              </button>
            ))}
          </div>
        </aside>
      </main>
      <form className="eaa-input-panel" onSubmit={onSubmit}>
        {processing ? <div className="eaa-processing">Agent is processing...</div> : null}
        <div className="eaa-input-row">
          <textarea
            ref={inputRef}
            className="eaa-input"
            rows={3}
            placeholder="Type a message. Paste an image or use <img /absolute/path/to/image.png>."
            value={content}
            onBlur={() => window.setTimeout(() => setSuggestionsOpen(false), 150)}
            onChange={(event) => void onInputChange(event.target.value)}
            onKeyDown={onKeyDown}
            onPaste={onPaste}
          />
          {suggestionsOpen && skillMatches.length ? (
            <div className="eaa-skill-suggestions">
              {skillMatches.map((skill) => (
                <button className="eaa-skill-suggestion" key={skill.name} type="button" onMouseDown={(event) => chooseSkill(event, skill)}>
                  <span className="eaa-skill-name">{skill.name}</span>
                  <span className="eaa-skill-description" dangerouslySetInnerHTML={{ __html: escapeHtml(skill.description ?? "") }} />
                </button>
              ))}
            </div>
          ) : null}
          <button className="eaa-send" type="submit" disabled={processing}>
            <Send size={16} aria-hidden="true" />
            <span>Send</span>
          </button>
        </div>
      </form>
      {previewImage ? (
        <div className="eaa-browser-image-preview open" aria-hidden="false" onClick={() => setPreviewImage(null)}>
          <button className="eaa-browser-image-preview-close" type="button" aria-label="Close image preview">
            <X size={22} aria-hidden="true" />
          </button>
          <img className="eaa-browser-image-preview-image" src={previewImage} alt="" onClick={(event) => event.stopPropagation()} />
        </div>
      ) : null}
    </div>
  );
}

export default App;
