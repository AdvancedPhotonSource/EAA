import { FormEvent, KeyboardEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleStop, Send, X } from "lucide-react";

import { escapeHtml, renderMarkdown } from "./markdown";
import type {
  PendingApproval,
  RuntimeConversation,
  RuntimeLogEntry,
  RuntimeSnapshot,
  Skill,
  WebUIConfig,
  WebUIMessage,
} from "./types";
import "./styles.css";

type ConnectionState = "Connecting..." | "Connected" | "Reconnecting..." | "Interrupt requested";
type SlashSuggestion = {
  key: string;
  name: string;
  detail: string;
  value: string;
};

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

const slashCommands: SlashSuggestion[] = [
  {
    key: "/exit",
    name: "/exit",
    detail: "Exit the current loop. Usage: /exit",
    value: "/exit",
  },
  {
    key: "/chat",
    name: "/chat",
    detail: "Enter chat mode. Usage: /chat",
    value: "/chat",
  },
  {
    key: "/monitor",
    name: "/monitor",
    detail: "Enter monitoring mode. Usage: /monitor <task description>",
    value: "/monitor ",
  },
  {
    key: "/skill",
    name: "/skill",
    detail: "Display or load agent skills. Usage: /skill [name]",
    value: "/skill ",
  },
  {
    key: "/setcodingtoolapproval",
    name: "/setcodingtoolapproval",
    detail: "Require or skip approval for Python and Bash coding tools. Usage: /setcodingtoolapproval true|false",
    value: "/setcodingtoolapproval ",
  },
  {
    key: "/setcodingtoolsandboxtype",
    name: "/setcodingtoolsandboxtype",
    detail: "Set the coding tool sandbox. Usage: /setcodingtoolsandboxtype none|bubblewrap|container [visible_dir ...]",
    value: "/setcodingtoolsandboxtype ",
  },
  {
    key: "/return",
    name: "/return",
    detail: "Return to the upper-level task. Usage: /return",
    value: "/return",
  },
];

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

const logKey = (log: RuntimeLogEntry, index: number) => String(log.id ?? `log-${index}`);

const formatLogTime = (timestamp: string) => {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};

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
  const [conversations, setConversations] = useState<RuntimeConversation[]>([
    { id: "primary", label: "Primary", kind: "primary", status: "idle", terminated: false, messages: [] },
  ]);
  const [activeConversationId, setActiveConversationId] = useState("primary");
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
  const [logs, setLogs] = useState<RuntimeLogEntry[]>([]);
  const messagesRef = useRef<HTMLDivElement>(null);
  const imagesRef = useRef<HTMLDivElement>(null);
  const logsRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const renderedIdsRef = useRef<Set<string>>(new Set());
  const pendingMessagesRef = useRef<Map<string, string>>(new Map());
  const infoTimeoutRef = useRef<number | null>(null);

  const processing = !inputRequested && status !== "idle";
  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId) ?? conversations[0];
  const activeMessages = activeConversation?.messages ?? [];

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

  const registerImages = useCallback((_sources: string[]) => {}, []);

  useEffect(() => {
    followScroll(messagesRef.current);
  }, [activeMessages, activeConversationId, followScroll]);

  useEffect(() => {
    followScroll(imagesRef.current);
  }, [sidebarImages, followScroll]);

  useEffect(() => {
    followScroll(logsRef.current);
  }, [logs, followScroll]);

  useEffect(() => {
    if (!window.MathJax?.typesetPromise || !messagesRef.current) return;
    window.MathJax.typesetPromise([messagesRef.current]).catch((error) => console.warn("MathJax typesetting failed:", error));
  }, [activeMessages]);

  useEffect(() => {
    const sources: string[] = [];
    const seen = new Set<string>();
    for (const message of activeMessages) {
      const role = String(message.role ?? "message");
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
    }
    setSidebarImages(sources);
  }, [activeMessages]);

  const upsertConversation = useCallback((conversation: RuntimeConversation, select = false) => {
    setConversations((previous) => {
      const index = previous.findIndex((item) => item.id === conversation.id);
      if (index === -1) return [...previous, { ...conversation, messages: conversation.messages ?? [] }];
      const next = [...previous];
      next[index] = {
        ...next[index],
        ...conversation,
        messages: next[index].messages ?? conversation.messages ?? [],
      };
      return next;
    });
    if (select) setActiveConversationId(conversation.id);
  }, []);

  const consumePending = useCallback(
    (message: WebUIMessage, conversationId = "primary") => {
      const role = message.role;
      if (!["user", "user_webui"].includes(String(role))) return false;
      const trimmed = String(message.content ?? "").trim();
      for (const [id, pendingContent] of pendingMessagesRef.current.entries()) {
        if (id.startsWith(`${conversationId}:`) && pendingContent === trimmed) {
          pendingMessagesRef.current.delete(id);
          return true;
        }
      }
      return false;
    },
    [],
  );

  const mergeMessages = useCallback(
    (incoming: WebUIMessage[], conversationId = "primary") => {
      const toRender: WebUIMessage[] = [];
      incoming.forEach((message, index) => {
        const id = `${conversationId}:${messageKey(message, index)}`;
        if (renderedIdsRef.current.has(id)) return;
        renderedIdsRef.current.add(id);
        if (!consumePending(message, conversationId)) toRender.push(message);
      });
      if (toRender.length) {
        setConversations((previous) => {
          if (!previous.some((conversation) => conversation.id === conversationId)) {
            return [
              ...previous,
              {
                id: conversationId,
                label: conversationId.replace(/-/g, " "),
                kind: conversationId === "primary" ? "primary" : "subagent",
                messages: toRender,
              },
            ];
          }
          return previous.map((conversation) =>
            conversation.id === conversationId
              ? { ...conversation, messages: [...(conversation.messages ?? []), ...toRender] }
              : conversation,
          );
        });
      }
    },
    [consumePending],
  );

  const applyStatus = useCallback((payload: RuntimeSnapshot) => {
    setStatus(payload.status ?? "idle");
    setInputRequested(Boolean(payload.input_requested));
    setInterruptRequested(Boolean(payload.interrupt_requested));
    if (payload.interrupt_requested) setConnection("Interrupt requested");
    const conversation = (payload as RuntimeSnapshot & { conversation?: RuntimeConversation }).conversation;
    if (conversation) upsertConversation(conversation);
  }, [upsertConversation]);

  const renderApprovalRequest = useCallback((payload: PendingApproval) => {
    const conversationId = payload.conversation_id || "primary";
    const approvalContent = `Tool '${payload.tool_name || "tool"}' requires approval before execution.\nArguments: ${JSON.stringify(
      payload.arguments || {},
      null,
      2,
    )}\nApprove? [y/N]: `;
    mergeMessages([{ id: `approval-${conversationId}-${Date.now()}`, role: "system", content: approvalContent }], conversationId);
  }, [mergeMessages]);

  const loadState = useCallback(async () => {
    try {
      const response = await fetch(config.routes.state);
      if (!response.ok) throw new Error("State fetch failed");
      const payload = (await response.json()) as RuntimeSnapshot;
      if (Array.isArray(payload.conversations) && payload.conversations.length) {
        setConversations(payload.conversations.map((conversation) => ({ ...conversation, messages: conversation.messages ?? [] })));
        if (!payload.conversations.some((conversation) => conversation.id === activeConversationId)) {
          setActiveConversationId(payload.conversations[0].id);
        }
      } else {
        mergeMessages(Array.isArray(payload.messages) ? payload.messages : [], "primary");
      }
      applyStatus(payload);
      if (Array.isArray(payload.logs)) setLogs(payload.logs);
      for (const conversation of payload.conversations ?? []) {
        if (conversation.pending_approval) renderApprovalRequest({ ...conversation.pending_approval, conversation_id: conversation.id });
      }
      if (!payload.conversations?.length && payload.pending_approval) renderApprovalRequest(payload.pending_approval);
      setConnection("Connected");
    } catch (_error) {
      setConnection("Reconnecting...");
    }
  }, [activeConversationId, applyStatus, mergeMessages, renderApprovalRequest]);

  useEffect(() => {
    loadState();
  }, [loadState]);

  useEffect(() => {
    const source = new EventSource(config.routes.events);
    source.onopen = () => setConnection("Connected");
    source.onerror = () => setConnection("Reconnecting...");
    source.addEventListener("message.created", (event) => {
      const payload = JSON.parse(event.data || "{}") as { conversation_id?: string; message?: WebUIMessage };
      if (payload.message) mergeMessages([payload.message], payload.conversation_id || "primary");
    });
    source.addEventListener("conversation.created", (event) => {
      const payload = JSON.parse(event.data || "{}") as { conversation?: RuntimeConversation };
      if (payload.conversation) upsertConversation(payload.conversation, true);
    });
    source.addEventListener("conversation.terminated", (event) => {
      const payload = JSON.parse(event.data || "{}") as { conversation?: RuntimeConversation };
      if (payload.conversation) upsertConversation(payload.conversation);
    });
    source.addEventListener("status.changed", (event) => applyStatus(JSON.parse(event.data || "{}") as RuntimeSnapshot));
    source.addEventListener("interrupt.requested", (event) => applyStatus(JSON.parse(event.data || "{}") as RuntimeSnapshot));
    source.addEventListener("interrupt.cleared", (event) => applyStatus(JSON.parse(event.data || "{}") as RuntimeSnapshot));
    source.addEventListener("input.requested", () => {
      setStatus("waiting_for_input");
      setInputRequested(true);
    });
    source.addEventListener("approval.requested", (event) => renderApprovalRequest(JSON.parse(event.data || "{}") as PendingApproval));
    source.addEventListener("log.created", (event) => {
      const payload = JSON.parse(event.data || "{}") as { log?: RuntimeLogEntry };
      if (payload.log) setLogs((previous) => [...previous, payload.log as RuntimeLogEntry].slice(-500));
    });
    return () => source.close();
  }, [applyStatus, mergeMessages, renderApprovalRequest, upsertConversation]);

  const submitApproval = async (approved: boolean, conversationId = activeConversationId) => {
    await fetch(config.routes.approval, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId, approved }),
    });
  };

  const appendPendingMessage = (pendingContent: string) => {
    const id = `pending-${pendingCounter}`;
    const scopedId = `primary:${id}`;
    setPendingCounter((value) => value + 1);
    pendingMessagesRef.current.set(scopedId, pendingContent.trim());
    setConversations((previous) =>
      previous.map((conversation) =>
        conversation.id === "primary"
          ? { ...conversation, messages: [...(conversation.messages ?? []), { id, role: "user", content: pendingContent, images: [], pending: true }] }
          : conversation,
      ),
    );
    renderedIdsRef.current.add(scopedId);
    return id;
  };

  const removePendingMessage = (id: string) => {
    pendingMessagesRef.current.delete(`primary:${id}`);
    renderedIdsRef.current.delete(`primary:${id}`);
    setConversations((previous) =>
      previous.map((conversation) =>
        conversation.id === "primary"
          ? { ...conversation, messages: (conversation.messages ?? []).filter((message) => message.id !== id) }
          : conversation,
      ),
    );
  };

  const showInfoMessage = (text: string) => {
    if (infoTimeoutRef.current !== null) window.clearTimeout(infoTimeoutRef.current);
    setInfoMessage({ id: Date.now(), text });
    infoTimeoutRef.current = window.setTimeout(() => {
      setInfoMessage(null);
      infoTimeoutRef.current = null;
    }, 5500);
  };

  const sendCurrentMessage = async () => {
    if (processing) return;
    const trimmed = content.trim();
    if (!trimmed) return;
    const pendingId = status === "waiting_for_approval" ? null : appendPendingMessage(trimmed);
    const response = await fetch(config.routes.send, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: trimmed }),
    });
    const payload = (await response.json().catch(() => ({}))) as { handled_as?: string; message?: string };
    if (response.status === 409) {
      if (pendingId !== null) removePendingMessage(pendingId);
      showInfoMessage(payload.message || "Please avoid sending repeated messages.");
      return;
    }
    if (!response.ok) {
      if (pendingId !== null) removePendingMessage(pendingId);
      throw new Error("Send failed");
    }
    if (payload.handled_as === "approval") {
      if (pendingId !== null) removePendingMessage(pendingId);
      setContent("");
      setSuggestionsOpen(false);
      return;
    }
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
    showInfoMessage("Workflow will be interrupted after the next LLM or tool response.");
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
    if (!value.startsWith("/")) {
      setSuggestionsOpen(false);
      return;
    }
    if (value.startsWith("/skill ")) await refreshSkills();
    setSuggestionsOpen(true);
  };

  const slashSuggestions = useMemo(() => {
    if (!content.startsWith("/")) return [];
    if (content.startsWith("/skill ")) {
      const typed = content.slice("/skill".length).trimStart().toLowerCase();
      return skills
        .filter((skill) => !typed || String(skill.name ?? "").toLowerCase().startsWith(typed))
        .slice(0, 8)
        .map((skill) => ({
          key: `/skill ${skill.name ?? ""}`,
          name: `/skill ${skill.name ?? ""}`,
          detail: skill.description ? `Load skill. ${skill.description}` : "Load this skill into the next model context.",
          value: `/skill ${skill.name ?? ""} `,
        }));
    }
    const commandToken = content.split(/\s/, 1)[0].toLowerCase();
    return slashCommands.filter((command) => command.name.startsWith(commandToken)).slice(0, 8);
  }, [content, skills]);

  const chooseSlashSuggestion = (event: MouseEvent<HTMLButtonElement>, suggestion: SlashSuggestion) => {
    event.preventDefault();
    setContent(suggestion.value);
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
          <div className="eaa-tabs" role="tablist" aria-label="Conversations">
            {conversations.map((conversation) => {
              const hasApproval = Boolean(conversation.pending_approval);
              const active = conversation.id === activeConversationId;
              return (
                <button
                  className={`eaa-tab${active ? " eaa-tab-active" : ""}${hasApproval ? " eaa-tab-alert" : ""}`}
                  key={conversation.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setActiveConversationId(conversation.id)}
                >
                  <span>{conversation.label || conversation.id}</span>
                  {hasApproval ? <span className="eaa-tab-badge">Approval</span> : null}
                </button>
              );
            })}
          </div>
          <div className="eaa-messages" ref={messagesRef} onScroll={registerScrollIntent}>
            {activeMessages.map((message, index) => (
              <MessageView
                key={messageKey(message, index)}
                message={message}
                onImage={setPreviewImage}
                onApproval={(approved) => submitApproval(approved, activeConversationId)}
                registerImages={registerImages}
              />
            ))}
            {activeConversation?.terminated ? <div className="eaa-termination-marker">Subagent terminated</div> : null}
          </div>
          <form className="eaa-input-panel" onSubmit={onSubmit}>
            {processing ? (
              <div className="eaa-processing" aria-label="Agent is processing" role="status">
                <span className="eaa-processing-dot" />
                <span className="eaa-processing-dot" />
                <span className="eaa-processing-dot" />
              </div>
            ) : null}
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
              {suggestionsOpen && slashSuggestions.length ? (
                <div className="eaa-slash-suggestions">
                  {slashSuggestions.map((suggestion) => (
                    <button
                      className="eaa-slash-suggestion"
                      key={suggestion.key}
                      type="button"
                      onMouseDown={(event) => chooseSlashSuggestion(event, suggestion)}
                    >
                      <span className="eaa-slash-name">{suggestion.name}</span>
                      <span className="eaa-slash-description" dangerouslySetInnerHTML={{ __html: escapeHtml(suggestion.detail) }} />
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
        </section>
        <aside className="eaa-sidebar" aria-label="Images and logs">
          <div className="eaa-side-panel eaa-image-panel">
            <div className="eaa-sidebar-title">Images</div>
            <div className="eaa-images" ref={imagesRef} onScroll={registerScrollIntent}>
              {sidebarImages.map((source) => (
                <button className="eaa-image-button" key={source} type="button" onClick={() => setPreviewImage(source)}>
                  <img className="eaa-sidebar-image" src={source} loading="lazy" decoding="async" alt="" />
                </button>
              ))}
            </div>
          </div>
          <div className="eaa-side-panel eaa-log-panel">
            <div className="eaa-sidebar-title">Logs</div>
            <div className="eaa-logs" ref={logsRef} onScroll={registerScrollIntent}>
              {logs.map((log, index) => (
                <div className={`eaa-log-entry eaa-log-${String(log.level || "info").toLowerCase()}`} key={logKey(log, index)}>
                  <div className="eaa-log-meta">
                    <span>{formatLogTime(log.timestamp)}</span>
                    <span>{log.source}</span>
                    <span>{log.level}</span>
                    {log.tool_name ? <span>{log.tool_name}</span> : null}
                  </div>
                  <div className="eaa-log-message">{log.message}</div>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>
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
