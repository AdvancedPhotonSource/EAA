import { CSSProperties, FormEvent, KeyboardEvent, MouseEvent, PointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  ChartNoAxesCombined,
  CircleStop,
  HelpCircle,
  Image as ImageIcon,
  Maximize2,
  MessageCircle,
  Paperclip,
  RefreshCw,
  Send,
  Settings,
  TerminalSquare,
  Wrench,
  X,
} from "lucide-react";

import { escapeHtml, renderMarkdown } from "./markdown";
import type {
  PendingApproval,
  RuntimeConversation,
  RuntimeLogEntry,
  RuntimeSnapshot,
  RuntimeVisualizationTile,
  Skill,
  ToolSchema,
  WebUIConfig,
  WebUIMessage,
} from "./types";
import "./styles.css";

type ConnectionState = "Connecting..." | "Connected" | "Reconnecting..." | "Interrupt requested";
type ViewName = "chat" | "visualizations" | "tools" | "settings";
type SlashSuggestion = {
  key: string;
  name: string;
  detail: string;
  value: string;
};
type ImageItem = {
  source: string;
  title: string;
  messageDomId: string;
};
type MCPReconnectStatus = {
  state: "success" | "error";
  message: string;
};
type PanelBox = {
  start: number;
  size: number;
};

const defaultConfig: WebUIConfig = {
  title: "EAA Control Center",
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
    toolSchemas: "/api/tool-schemas",
    mcpReconnect: "/api/mcp/reconnect",
    mathjax: "/static/mathjax",
  },
};

const config = window.EAA_WEBUI_CONFIG ?? defaultConfig;
const DEFAULT_CHAT_PANEL_PERCENT = 57;
const DEFAULT_IMAGE_PANEL_PERCENT = 41;
const MAIN_RESIZER_SIZE_PX = 16;
const SIDEBAR_RESIZER_SIZE_PX = 16;
const MIN_CHAT_PANEL_PX = 360;
const MIN_SIDEBAR_PANEL_PX = 300;
const MIN_IMAGE_PANEL_PX = 140;
const MIN_LOG_PANEL_PX = 180;

const mathJaxScriptUrl = () => `${config.routes.mathjax.replace(/\/$/, "")}/es5/tex-svg-full.js`;

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const contentBox = (element: HTMLElement, axis: "horizontal" | "vertical"): PanelBox => {
  const rect = element.getBoundingClientRect();
  const styles = window.getComputedStyle(element);
  if (axis === "horizontal") {
    const paddingStart = Number.parseFloat(styles.paddingLeft) || 0;
    const paddingEnd = Number.parseFloat(styles.paddingRight) || 0;
    return { start: rect.left + paddingStart, size: rect.width - paddingStart - paddingEnd };
  }
  const paddingStart = Number.parseFloat(styles.paddingTop) || 0;
  const paddingEnd = Number.parseFloat(styles.paddingBottom) || 0;
  return { start: rect.top + paddingStart, size: rect.height - paddingStart - paddingEnd };
};

const clampPanelPercent = (percent: number, size: number, firstMin: number, secondMin: number, resizerSize: number) => {
  if (size <= firstMin + secondMin + resizerSize) return clamp(percent, 0, 100);
  const min = (firstMin / size) * 100;
  const max = ((size - secondMin - resizerSize) / size) * 100;
  return clamp(percent, min, max);
};

const configureMathJax = () => {
  const current = window.MathJax ?? {};
  window.MathJax = {
    ...current,
    tex: {
      ...(current.tex ?? {}),
      inlineMath: [["\\(", "\\)"]],
      displayMath: [
        ["$$", "$$"],
        ["\\[", "\\]"],
      ],
      processEscapes: true,
    },
    svg: {
      ...(current.svg ?? {}),
      fontCache: "global",
    },
  };
};

const ensureMathJaxScript = () => {
  const existing = Array.from(document.scripts).find((script) => script.dataset.eaaMathjax === "true");
  if (existing) return existing;
  const script = document.createElement("script");
  script.src = mathJaxScriptUrl();
  script.defer = true;
  script.dataset.eaaMathjax = "true";
  script.addEventListener("error", () => console.warn(`MathJax failed to load from ${mathJaxScriptUrl()}`));
  document.head.appendChild(script);
  return script;
};

const slashCommands: SlashSuggestion[] = [
  { key: "/exit", name: "/exit", detail: "Exit the current loop. Usage: /exit", value: "/exit" },
  { key: "/chat", name: "/chat", detail: "Enter chat mode. Usage: /chat", value: "/chat" },
  { key: "/monitor", name: "/monitor", detail: "Enter monitoring mode. Usage: /monitor <task description>", value: "/monitor " },
  { key: "/skill", name: "/skill", detail: "Display or load agent skills. Usage: /skill [name]", value: "/skill " },
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
  { key: "/return", name: "/return", detail: "Return to the upper-level task. Usage: /return", value: "/return" },
];

const roleLabel = (role: string) => (role === "user_webui" || role === "user" ? "You" : role === "assistant" ? "EAA Main Agent" : role);

const roleAvatarLetter = (role: string) => {
  if (role === "assistant") return "A";
  if (role === "system") return "S";
  if (role === "tool") return "T";
  if (role === "user" || role === "user_webui") return "Y";
  return role.slice(0, 1).toUpperCase() || "M";
};

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

const messageDomId = (conversationId: string, message: WebUIMessage, index: number) =>
  `message-${conversationId}-${String(message.id ?? index).replace(/[^a-zA-Z0-9_-]/g, "-")}`;

const messageContentKey = (message: WebUIMessage) => `${message.role ?? ""}:${String(message.content ?? "").trim()}`;

const stringHash = (value: string) => {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash.toString(36);
};

const imagePathTitle = (pathOrSource: unknown) => {
  const raw = String(pathOrSource ?? "");
  if (!raw || raw.startsWith("data:image")) return null;
  const path = raw.startsWith("/api/image") ? decodeURIComponent(raw.split("path=")[1] || "") : raw;
  const name = path.split(/[\\/]/).filter(Boolean).pop();
  return name || null;
};

const isApprovalMessage = (message: WebUIMessage) =>
  String(message.role ?? "") === "system" && /Approve\?\s*\[y\/N\]:/i.test(String(message.content ?? ""));

const formatRemainingTime = (milliseconds: number) => {
  const seconds = Math.max(0, Math.ceil(milliseconds / 1000));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
};

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

const imageTimestampTitle = (message: WebUIMessage) => {
  if (!message.timestamp) return null;
  return formatLogTime(message.timestamp) || null;
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
  conversationId,
  index,
  message,
  onImage,
  onApproval,
}: {
  conversationId: string;
  index: number;
  message: WebUIMessage;
  onImage: (src: string) => void;
  onApproval: (approved: boolean, approvalId?: string) => Promise<void>;
}) {
  const [approvalSubmitted, setApprovalSubmitted] = useState(false);
  const [approvalRemainingMs, setApprovalRemainingMs] = useState<number | null>(null);
  const role = String(message.role ?? "message");
  const content = String(message.content ?? "").trim();
  const approvalExpiresAt = useMemo(() => {
    if (message.approval_expires_at) return Date.parse(message.approval_expires_at);
    if (!message.approval_timeout_seconds) return NaN;
    const requestedAt = message.approval_requested_at ? Date.parse(message.approval_requested_at) : Date.now();
    return requestedAt + message.approval_timeout_seconds * 1000;
  }, [message.approval_expires_at, message.approval_requested_at, message.approval_timeout_seconds]);
  const approvalExpired = approvalRemainingMs !== null && approvalRemainingMs <= 0;
  const messageTime = message.timestamp ? formatLogTime(message.timestamp) : "";
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
    if (!isApprovalMessage(message) || Number.isNaN(approvalExpiresAt)) {
      setApprovalRemainingMs(null);
      return;
    }
    const updateRemaining = () => setApprovalRemainingMs(Math.max(0, approvalExpiresAt - Date.now()));
    updateRemaining();
    const interval = window.setInterval(updateRemaining, 1000);
    return () => window.clearInterval(interval);
  }, [approvalExpiresAt, message]);

  const submitApproval = async (approved: boolean) => {
    if (approvalSubmitted || approvalExpired) return;
    setApprovalSubmitted(true);
    await onApproval(approved, message.approval_id);
  };

  return (
    <article className={`eaa-message eaa-message-${role}`} id={messageDomId(conversationId, message, index)}>
      <div className="eaa-message-meta">
        <div className={`eaa-avatar eaa-avatar-${role}`}>{roleAvatarLetter(role)}</div>
        <div className="eaa-role">{roleLabel(role)}</div>
        {messageTime ? (
          <time className="eaa-message-time" dateTime={message.timestamp}>
            {messageTime}
          </time>
        ) : null}
      </div>
      <div className="eaa-message-body">
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
          <>
            {approvalRemainingMs !== null ? (
              <div className={`eaa-approval-timer${approvalExpired ? " eaa-approval-timer-expired" : ""}`}>
                {approvalExpired ? "Approval timed out and was rejected." : `Auto-rejects in ${formatRemainingTime(approvalRemainingMs)}`}
              </div>
            ) : null}
            <div className="eaa-approval-actions">
              <button
                className="eaa-approval-button eaa-approval-yes"
                disabled={approvalSubmitted || approvalExpired}
                type="button"
                onClick={() => submitApproval(true)}
              >
                Yes
              </button>
              <button
                className="eaa-approval-button eaa-approval-no"
                disabled={approvalSubmitted || approvalExpired}
                type="button"
                onClick={() => submitApproval(false)}
              >
                No
              </button>
            </div>
          </>
        ) : null}
      </div>
    </article>
  );
}

function ConversationTabs({
  conversations,
  activeConversationId,
  onSelect,
  onClose,
}: {
  conversations: RuntimeConversation[];
  activeConversationId: string;
  onSelect: (conversationId: string) => void;
  onClose: (conversationId: string) => void;
}) {
  return (
    <div className="eaa-tabs" role="tablist" aria-label="Conversations">
      {conversations.map((conversation) => {
        const hasApproval = Boolean(conversation.pending_approval);
        const active = conversation.id === activeConversationId;
        const closable = conversation.id !== "primary" && conversation.kind !== "primary";
        return (
          <div
            className={`eaa-tab${active ? " eaa-tab-active" : ""}${hasApproval ? " eaa-tab-alert" : ""}`}
            key={conversation.id}
            role="tab"
            aria-selected={active}
          >
            <button className="eaa-tab-select" type="button" onClick={() => onSelect(conversation.id)}>
              <span>{conversation.kind === "primary" ? "Main Agent" : conversation.label || conversation.id}</span>
            </button>
            {hasApproval ? <span className="eaa-tab-badge">Approval</span> : null}
            {closable ? (
              <button
                className="eaa-tab-close"
                type="button"
                aria-label={`Close ${conversation.label || conversation.id}`}
                onClick={() => onClose(conversation.id)}
              >
                <X size={14} aria-hidden="true" />
              </button>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function VisualizationTileView({ tile, onImage }: { tile: RuntimeVisualizationTile; onImage: (src: string) => void }) {
  const imagePath = tile.content?.type === "image" ? tile.content.image_path : undefined;
  const source = imagePath ? imageSource(imagePath) : null;
  const width = Math.max(180, Math.round(Number(tile.width) || 0));
  const height = Math.max(140, Math.round(Number(tile.height) || 0));
  return (
    <article className="eaa-visualization-tile" style={{ width, height }}>
      {source ? (
        <button className="eaa-visualization-image-button" type="button" onClick={() => onImage(source)}>
          <img className="eaa-visualization-image" src={source} loading="lazy" decoding="async" alt="" />
        </button>
      ) : (
        <div className="eaa-visualization-placeholder">Waiting for plot</div>
      )}
    </article>
  );
}

function VisualizationsView({
  conversations,
  activeConversationId,
  activeConversation,
  onSelectConversation,
  onCloseConversation,
  onImage,
}: {
  conversations: RuntimeConversation[];
  activeConversationId: string;
  activeConversation?: RuntimeConversation;
  onSelectConversation: (conversationId: string) => void;
  onCloseConversation: (conversationId: string) => void;
  onImage: (src: string) => void;
}) {
  const tiles = activeConversation?.visualization_tiles ?? [];
  return (
    <main className="eaa-visualization-main">
      <section className="eaa-visualizations" aria-label="Visualizations">
        <ConversationTabs
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelect={onSelectConversation}
          onClose={onCloseConversation}
        />
        <div className="eaa-visualization-canvas">
          {tiles.map((tile) => (
            <VisualizationTileView key={tile.id} tile={tile} onImage={onImage} />
          ))}
          {!tiles.length ? <div className="eaa-empty-state">No visualization tiles yet.</div> : null}
        </div>
      </section>
    </main>
  );
}

function ToolsView({
  tools,
  reconnectingMcpServers,
  mcpReconnectStatuses,
  onReconnectMcp,
}: {
  tools: ToolSchema[];
  reconnectingMcpServers: Set<string>;
  mcpReconnectStatuses: Record<string, MCPReconnectStatus>;
  onReconnectMcp: (serverId: string) => void;
}) {
  return (
    <section className="eaa-view eaa-tools-view" aria-label="Registered tools">
      <div className="eaa-view-header">
        <h2>Tools</h2>
        <span>{tools.length} registered</span>
      </div>
      <div className="eaa-tool-list">
        {tools.map((tool, index) => {
          const fn = tool.function ?? {};
          const parameters = fn.parameters ?? {};
          const properties = parameters.properties ?? {};
          const required = new Set(parameters.required ?? []);
          const mcpServerId = tool.mcp?.server_id?.trim();
          const mcpServerName = tool.mcp?.server_name || mcpServerId || "MCP server";
          const mcpReconnectStatus = mcpServerId ? mcpReconnectStatuses[mcpServerId] : undefined;
          const mcpReconnectInProgress = mcpServerId ? reconnectingMcpServers.has(mcpServerId) : false;
          return (
            <article className="eaa-tool-card" key={`${fn.name ?? "tool"}-${index}`}>
              <div className="eaa-tool-card-header">
                <div>
                  <h3>{fn.name || `Tool ${index + 1}`}</h3>
                  {tool.mcp?.server_name ? <span className="eaa-tool-source">MCP: {tool.mcp.server_name}</span> : null}
                </div>
                {mcpServerId ? (
                  <button
                    className={`eaa-tool-action${mcpReconnectInProgress ? " eaa-tool-action-busy" : ""}`}
                    type="button"
                    disabled={mcpReconnectInProgress}
                    title={`Reconnect ${mcpServerName}`}
                    aria-label={`Reconnect ${mcpServerName}`}
                    onClick={() => onReconnectMcp(mcpServerId)}
                  >
                    <RefreshCw size={15} aria-hidden="true" />
                    <span>{mcpReconnectInProgress ? "Reconnecting" : "Reconnect"}</span>
                  </button>
                ) : null}
              </div>
              <p>{fn.description || "No description provided."}</p>
              {mcpReconnectStatus ? (
                <div className={`eaa-tool-status eaa-tool-status-${mcpReconnectStatus.state}`}>
                  {mcpReconnectStatus.message}
                </div>
              ) : null}
              <div className="eaa-argument-table">
                {Object.keys(properties).length ? (
                  Object.entries(properties).map(([name, schema]) => (
                    <div className="eaa-argument-row" key={name}>
                      <div>
                        <strong>{name}</strong>
                        {required.has(name) ? <span>Required</span> : null}
                      </div>
                      <pre>{JSON.stringify(schema, null, 2)}</pre>
                    </div>
                  ))
                ) : (
                  <div className="eaa-empty-inline">No arguments.</div>
                )}
              </div>
            </article>
          );
        })}
        {!tools.length ? <div className="eaa-empty-state">No tools are registered with the agent.</div> : null}
      </div>
    </section>
  );
}

function SettingsView({ title, onTitleChange }: { title: string; onTitleChange: (title: string) => void }) {
  return (
    <section className="eaa-view eaa-settings-view" aria-label="Settings">
      <div className="eaa-view-header">
        <h2>Settings</h2>
      </div>
      <label className="eaa-setting-field">
        <span>WebUI title</span>
        <input value={title} onChange={(event) => onTitleChange(event.target.value)} />
      </label>
    </section>
  );
}

function HelpDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="eaa-modal-backdrop" onClick={onClose}>
      <div className="eaa-help-dialog" role="dialog" aria-modal="true" aria-label="Help" onClick={(event) => event.stopPropagation()}>
        <div className="eaa-dialog-header">
          <h2>Slash Commands</h2>
          <button type="button" aria-label="Close help" onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="eaa-command-list">
          {slashCommands.map((command) => (
            <div className="eaa-command-row" key={command.key}>
              <code>{command.name}</code>
              <span>{command.detail}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ImageGalleryDialog({
  images,
  onClose,
  onPreview,
  onJumpToMessage,
}: {
  images: ImageItem[];
  onClose: () => void;
  onPreview: (source: string) => void;
  onJumpToMessage: (messageId: string) => void;
}) {
  return (
    <div className="eaa-modal-backdrop" onClick={onClose}>
      <div className="eaa-gallery-dialog" role="dialog" aria-modal="true" aria-label="Image gallery" onClick={(event) => event.stopPropagation()}>
        <div className="eaa-dialog-header">
          <h2>Image Gallery</h2>
          <button type="button" aria-label="Close image gallery" onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="eaa-gallery-grid">
          {images.map((image) => (
            <article className="eaa-gallery-card" key={image.source}>
              <button className="eaa-gallery-image-button" type="button" onClick={() => onPreview(image.source)}>
                <img src={image.source} loading="lazy" decoding="async" alt="" />
              </button>
              <button className="eaa-gallery-title" type="button" onClick={() => onJumpToMessage(image.messageDomId)}>
                {image.title}
              </button>
            </article>
          ))}
          {!images.length ? <div className="eaa-empty-state">No images yet.</div> : null}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [conversations, setConversations] = useState<RuntimeConversation[]>([
    { id: "primary", label: "Primary", kind: "primary", status: "idle", terminated: false, messages: [], visualization_tiles: [] },
  ]);
  const [activeConversationId, setActiveConversationId] = useState("primary");
  const [activeView, setActiveView] = useState<ViewName>("chat");
  const [pendingCounter, setPendingCounter] = useState(0);
  const [connection, setConnection] = useState<ConnectionState>("Connecting...");
  const [status, setStatus] = useState("idle");
  const [inputRequested, setInputRequested] = useState(false);
  const [interruptRequested, setInterruptRequested] = useState(false);
  const [infoMessage, setInfoMessage] = useState<{ id: number; text: string } | null>(null);
  const [content, setContent] = useState("");
  const [skills, setSkills] = useState<Skill[]>([]);
  const [skillsLoaded, setSkillsLoaded] = useState(false);
  const [toolSchemas, setToolSchemas] = useState<ToolSchema[]>([]);
  const [reconnectingMcpServers, setReconnectingMcpServers] = useState<Set<string>>(new Set());
  const [mcpReconnectStatuses, setMcpReconnectStatuses] = useState<Record<string, MCPReconnectStatus>>({});
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [sidebarImages, setSidebarImages] = useState<ImageItem[]>([]);
  const [logs, setLogs] = useState<RuntimeLogEntry[]>([]);
  const [helpOpen, setHelpOpen] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [mathJaxReady, setMathJaxReady] = useState(Boolean(window.MathJax?.typesetPromise));
  const [uiTitle, setUiTitle] = useState(config.title);
  const [chatPanelPercent, setChatPanelPercent] = useState(DEFAULT_CHAT_PANEL_PERCENT);
  const [imagePanelPercent, setImagePanelPercent] = useState(DEFAULT_IMAGE_PANEL_PERCENT);
  const mainRef = useRef<HTMLElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const imagesRef = useRef<HTMLDivElement>(null);
  const logsRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const renderedIdsRef = useRef<Set<string>>(new Set());
  const pendingMessagesRef = useRef<Map<string, string>>(new Map());
  const closedConversationIdsRef = useRef<Set<string>>(new Set());
  const infoTimeoutRef = useRef<number | null>(null);

  const processing = !inputRequested && status !== "idle";
  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId) ?? conversations[0];
  const activeMessages = activeConversation?.messages ?? [];
  const chatPanelStyle = useMemo(
    () =>
      ({
        "--eaa-chat-panel-size": `${chatPanelPercent}%`,
        "--eaa-image-panel-size": `${imagePanelPercent}%`,
      }) as CSSProperties,
    [chatPanelPercent, imagePanelPercent],
  );

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

  const resizeChatPanel = useCallback((clientX: number) => {
    if (!mainRef.current) return;
    const box = contentBox(mainRef.current, "horizontal");
    const percent = ((clientX - box.start) / box.size) * 100;
    setChatPanelPercent(clampPanelPercent(percent, box.size, MIN_CHAT_PANEL_PX, MIN_SIDEBAR_PANEL_PX, MAIN_RESIZER_SIZE_PX));
  }, []);

  const resizeImagePanel = useCallback((clientY: number) => {
    if (!sidebarRef.current) return;
    const box = contentBox(sidebarRef.current, "vertical");
    const percent = ((clientY - box.start) / box.size) * 100;
    setImagePanelPercent(clampPanelPercent(percent, box.size, MIN_IMAGE_PANEL_PX, MIN_LOG_PANEL_PX, SIDEBAR_RESIZER_SIZE_PX));
  }, []);

  const startChatPanelResize = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      const target = event.currentTarget;
      const pointerId = event.pointerId;
      target.setPointerCapture(pointerId);
      document.body.classList.add("eaa-resizing-columns");
      resizeChatPanel(event.clientX);

      const onPointerMove = (moveEvent: globalThis.PointerEvent) => resizeChatPanel(moveEvent.clientX);
      const finishResize = () => {
        if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
        document.body.classList.remove("eaa-resizing-columns");
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", finishResize);
        window.removeEventListener("pointercancel", finishResize);
      };

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", finishResize);
      window.addEventListener("pointercancel", finishResize);
    },
    [resizeChatPanel],
  );

  const startImagePanelResize = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      const target = event.currentTarget;
      const pointerId = event.pointerId;
      target.setPointerCapture(pointerId);
      document.body.classList.add("eaa-resizing-rows");
      resizeImagePanel(event.clientY);

      const onPointerMove = (moveEvent: globalThis.PointerEvent) => resizeImagePanel(moveEvent.clientY);
      const finishResize = () => {
        if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
        document.body.classList.remove("eaa-resizing-rows");
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", finishResize);
        window.removeEventListener("pointercancel", finishResize);
      };

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", finishResize);
      window.addEventListener("pointercancel", finishResize);
    },
    [resizeImagePanel],
  );

  const adjustChatPanel = useCallback((delta: number) => {
    setChatPanelPercent((previous) => {
      if (!mainRef.current) return clamp(previous + delta, 0, 100);
      const box = contentBox(mainRef.current, "horizontal");
      return clampPanelPercent(previous + delta, box.size, MIN_CHAT_PANEL_PX, MIN_SIDEBAR_PANEL_PX, MAIN_RESIZER_SIZE_PX);
    });
  }, []);

  const adjustImagePanel = useCallback((delta: number) => {
    setImagePanelPercent((previous) => {
      if (!sidebarRef.current) return clamp(previous + delta, 0, 100);
      const box = contentBox(sidebarRef.current, "vertical");
      return clampPanelPercent(previous + delta, box.size, MIN_IMAGE_PANEL_PX, MIN_LOG_PANEL_PX, SIDEBAR_RESIZER_SIZE_PX);
    });
  }, []);

  const onChatPanelResizeKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      const step = event.shiftKey ? 5 : 2;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        adjustChatPanel(-step);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        adjustChatPanel(step);
      }
    },
    [adjustChatPanel],
  );

  const onImagePanelResizeKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      const step = event.shiftKey ? 5 : 2;
      if (event.key === "ArrowUp") {
        event.preventDefault();
        adjustImagePanel(-step);
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        adjustImagePanel(step);
      }
    },
    [adjustImagePanel],
  );

  useEffect(() => {
    document.title = uiTitle || config.title;
  }, [uiTitle]);

  useEffect(() => {
    followScroll(messagesRef.current);
  }, [activeMessages, activeConversationId, followScroll]);

  useEffect(() => {
    followScroll(logsRef.current);
  }, [logs, followScroll]);

  useEffect(() => {
    if (mathJaxReady || window.MathJax?.typesetPromise) {
      setMathJaxReady(true);
      return;
    }
    configureMathJax();
    ensureMathJaxScript();
    const interval = window.setInterval(() => {
      if (window.MathJax?.typesetPromise) {
        setMathJaxReady(true);
        window.clearInterval(interval);
      }
    }, 100);
    return () => window.clearInterval(interval);
  }, [mathJaxReady]);

  useEffect(() => {
    if (!mathJaxReady || !window.MathJax?.typesetPromise || !messagesRef.current) return;
    window.MathJax.typesetPromise([messagesRef.current]).catch((error) => console.warn("MathJax typesetting failed:", error));
  }, [activeMessages, mathJaxReady]);

  useEffect(() => {
    const items: ImageItem[] = [];
    const seen = new Set<string>();
    activeMessages.forEach((message, index) => {
      const role = String(message.role ?? "message");
      const attached = Array.isArray(message.images) && message.images.length ? message.images : message.image ? [message.image] : [];
      const addImage = (image: unknown) => {
        const source = imageSource(image);
        if (seen.has(source)) return;
        seen.add(source);
        const pathTitle = imagePathTitle(image);
        items.push({
          source,
          title: pathTitle ?? imageTimestampTitle(message) ?? "Image",
          messageDomId: messageDomId(activeConversationId, message, index),
        });
      };
      attached.forEach(addImage);
      if (role !== "system") parseContentImagePaths(message.content).forEach(addImage);
    });
    setSidebarImages(items);
  }, [activeMessages, activeConversationId]);

  const upsertConversation = useCallback((conversation: RuntimeConversation, select = false) => {
    if (conversation.id !== "primary" && closedConversationIdsRef.current.has(conversation.id)) return;
    setConversations((previous) => {
      const index = previous.findIndex((item) => item.id === conversation.id);
      if (index === -1) {
        return [
          ...previous,
          {
            ...conversation,
            messages: conversation.messages ?? [],
            visualization_tiles: conversation.visualization_tiles ?? [],
          },
        ];
      }
      const next = [...previous];
      const existingMessages = next[index].messages ?? [];
      const mergedMessages = [...existingMessages];
      const seenMessages = new Set(
        existingMessages.map((message, messageIndex) =>
          String(message.id ?? `${message.role ?? ""}:${message.content ?? ""}:${messageIndex}`),
        ),
      );
      const seenUserContent = new Set(
        existingMessages
          .filter((message) => ["user", "user_webui"].includes(String(message.role)))
          .map((message) => messageContentKey(message)),
      );
      for (const [messageIndex, message] of (conversation.messages ?? []).entries()) {
        const key = String(message.id ?? `${message.role ?? ""}:${message.content ?? ""}:${messageIndex}`);
        renderedIdsRef.current.add(`${conversation.id}:${messageKey(message, messageIndex)}`);
        if (seenMessages.has(key)) continue;
        if (["user", "user_webui"].includes(String(message.role)) && seenUserContent.has(messageContentKey(message))) continue;
        seenMessages.add(key);
        if (["user", "user_webui"].includes(String(message.role))) seenUserContent.add(messageContentKey(message));
        mergedMessages.push(message);
      }
      next[index] = {
        ...next[index],
        ...conversation,
        messages: mergedMessages,
        visualization_tiles: conversation.visualization_tiles ?? next[index].visualization_tiles ?? [],
      };
      return next;
    });
    if (select) setActiveConversationId(conversation.id);
  }, []);

  const consumePending = useCallback((message: WebUIMessage, conversationId = "primary") => {
    const role = message.role;
    if (!["user", "user_webui"].includes(String(role))) return false;
    const trimmed = String(message.content ?? "").trim();
    for (const [id, pendingContent] of pendingMessagesRef.current.entries()) {
      if (id.startsWith(`${conversationId}:`) && pendingContent === trimmed) {
        pendingMessagesRef.current.delete(id);
        const pendingId = id.slice(conversationId.length + 1);
        setConversations((previous) =>
          previous.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  messages: (conversation.messages ?? []).map((item) => (item.id === pendingId ? { ...message, pending: false } : item)),
                }
              : conversation,
          ),
        );
        return true;
      }
    }
    return false;
  }, []);

  const mergeMessages = useCallback(
    (incoming: WebUIMessage[], conversationId = "primary") => {
      if (conversationId !== "primary" && closedConversationIdsRef.current.has(conversationId)) return;
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
                visualization_tiles: [],
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

  const applyStatus = useCallback(
    (payload: RuntimeSnapshot) => {
      setStatus(payload.status ?? "idle");
      setInputRequested(Boolean(payload.input_requested));
      setInterruptRequested(Boolean(payload.interrupt_requested));
      if (payload.interrupt_requested) setConnection("Interrupt requested");
      const conversation = (payload as RuntimeSnapshot & { conversation?: RuntimeConversation }).conversation;
      if (conversation) upsertConversation(conversation);
    },
    [upsertConversation],
  );

  const renderApprovalRequest = useCallback(
    (payload: PendingApproval) => {
      const conversationId = payload.conversation_id || "primary";
      const approvalContent = `Tool '${payload.tool_name || "tool"}' requires approval before execution.\nArguments: ${JSON.stringify(
        payload.arguments || {},
        null,
        2,
      )}\nApprove? [y/N]: `;
      mergeMessages(
        [
          {
            id: payload.id || `approval-${conversationId}-${payload.tool_name || "tool"}-${stringHash(approvalContent)}`,
            role: "system",
            content: approvalContent,
            approval_id: payload.id,
            approval_requested_at: payload.requested_at,
            approval_expires_at: payload.expires_at,
            approval_timeout_seconds: payload.timeout_seconds,
          },
        ],
        conversationId,
      );
    },
    [mergeMessages],
  );

  const upsertVisualizationTile = useCallback((tile: RuntimeVisualizationTile, conversationId = "primary") => {
    if (conversationId !== "primary" && closedConversationIdsRef.current.has(conversationId)) return;
    setConversations((previous) => {
      const updateConversation = (conversation: RuntimeConversation) => {
        const tiles = conversation.visualization_tiles ?? [];
        const index = tiles.findIndex((item) => item.id === tile.id);
        const nextTiles = index === -1 ? [...tiles, tile] : tiles.map((item) => (item.id === tile.id ? tile : item));
        return { ...conversation, visualization_tiles: nextTiles };
      };
      if (!previous.some((conversation) => conversation.id === conversationId)) {
        return [
          ...previous,
          {
            id: conversationId,
            label: conversationId.replace(/-/g, " "),
            kind: conversationId === "primary" ? "primary" : "subagent",
            messages: [],
            visualization_tiles: [tile],
          },
        ];
      }
      return previous.map((conversation) => (conversation.id === conversationId ? updateConversation(conversation) : conversation));
    });
  }, []);

  const removeVisualizationTile = useCallback((tileId: string, conversationId = "primary") => {
    setConversations((previous) =>
      previous.map((conversation) =>
        conversation.id === conversationId
          ? {
              ...conversation,
              visualization_tiles: (conversation.visualization_tiles ?? []).filter((tile) => tile.id !== tileId),
            }
          : conversation,
      ),
    );
  }, []);

  const loadToolSchemas = useCallback(async () => {
    const response = await fetch(config.routes.toolSchemas);
    if (!response.ok) return;
    const payload = (await response.json()) as { tools?: ToolSchema[] };
    setToolSchemas(Array.isArray(payload.tools) ? payload.tools : []);
  }, []);

  const reconnectMcpServer = useCallback(
    async (serverId: string) => {
      if (!serverId || reconnectingMcpServers.has(serverId)) return;
      setReconnectingMcpServers((previous) => {
        const next = new Set(previous);
        next.add(serverId);
        return next;
      });
      setMcpReconnectStatuses((previous) => {
        const next = { ...previous };
        delete next[serverId];
        return next;
      });
      try {
        const response = await fetch(config.routes.mcpReconnect, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ server_id: serverId }),
        });
        const payload = (await response.json().catch(() => ({}))) as { error?: string; mcp?: { server_name?: string } };
        if (!response.ok) throw new Error(payload.error || "Reconnect failed");
        const serverName = payload.mcp?.server_name || "MCP server";
        setMcpReconnectStatuses((previous) => ({
          ...previous,
          [serverId]: { state: "success", message: `${serverName} reconnected.` },
        }));
        void loadToolSchemas();
      } catch (error) {
        setMcpReconnectStatuses((previous) => ({
          ...previous,
          [serverId]: {
            state: "error",
            message: error instanceof Error ? error.message : "Reconnect failed",
          },
        }));
      } finally {
        setReconnectingMcpServers((previous) => {
          const next = new Set(previous);
          next.delete(serverId);
          return next;
        });
      }
    },
    [loadToolSchemas, reconnectingMcpServers],
  );

  const loadState = useCallback(async () => {
    try {
      const response = await fetch(config.routes.state);
      if (!response.ok) throw new Error("State fetch failed");
      const payload = (await response.json()) as RuntimeSnapshot;
      if (Array.isArray(payload.conversations) && payload.conversations.length) {
        const visibleConversations = payload.conversations.filter(
          (conversation) => conversation.id === "primary" || !closedConversationIdsRef.current.has(conversation.id),
        );
        setConversations(
          visibleConversations.map((conversation) => ({
            ...conversation,
            messages: conversation.messages ?? [],
            visualization_tiles: conversation.visualization_tiles ?? [],
          })),
        );
        if (!visibleConversations.some((conversation) => conversation.id === activeConversationId)) {
          setActiveConversationId(visibleConversations[0]?.id ?? "primary");
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
    void loadToolSchemas();
  }, [loadState, loadToolSchemas]);

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
    source.addEventListener("visualization.tile.created", (event) => {
      const payload = JSON.parse(event.data || "{}") as { conversation_id?: string; tile?: RuntimeVisualizationTile };
      if (payload.tile) upsertVisualizationTile(payload.tile, payload.conversation_id || "primary");
    });
    source.addEventListener("visualization.tile.updated", (event) => {
      const payload = JSON.parse(event.data || "{}") as { conversation_id?: string; tile?: RuntimeVisualizationTile };
      if (payload.tile) upsertVisualizationTile(payload.tile, payload.conversation_id || "primary");
    });
    source.addEventListener("visualization.tile.removed", (event) => {
      const payload = JSON.parse(event.data || "{}") as { conversation_id?: string; tile_id?: string };
      if (payload.tile_id) removeVisualizationTile(payload.tile_id, payload.conversation_id || "primary");
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
  }, [applyStatus, mergeMessages, removeVisualizationTile, renderApprovalRequest, upsertConversation, upsertVisualizationTile]);

  const submitApproval = async (approved: boolean, conversationId = activeConversationId, approvalId?: string) => {
    await fetch(config.routes.approval, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId, approved, approval_id: approvalId }),
    });
  };

  const closeConversation = (conversationId: string) => {
    if (conversationId === "primary") return;
    closedConversationIdsRef.current.add(conversationId);
    setConversations((previous) => previous.filter((conversation) => conversation.id !== conversationId));
    if (activeConversationId === conversationId) setActiveConversationId("primary");
  };

  const appendPendingMessage = (pendingContent: string) => {
    const id = `pending-${pendingCounter}`;
    const scopedId = `primary:${id}`;
    setPendingCounter((value) => value + 1);
    pendingMessagesRef.current.set(scopedId, pendingContent.trim());
    setConversations((previous) =>
      previous.map((conversation) =>
        conversation.id === "primary" &&
        !(conversation.messages ?? []).some(
          (message) =>
            ["user", "user_webui"].includes(String(message.role)) &&
            String(message.content ?? "").trim() === pendingContent.trim(),
        )
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
    const activeApprovalPending = Boolean(activeConversation?.pending_approval) || status === "waiting_for_approval";
    if (activeApprovalPending && ["y", "yes", "n", "no"].includes(trimmed.toLowerCase())) {
      await submitApproval(["y", "yes"].includes(trimmed.toLowerCase()), activeConversationId);
      setContent("");
      setSuggestionsOpen(false);
      return;
    }
    const pendingId = activeApprovalPending ? null : appendPendingMessage(trimmed);
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

  const jumpToImageMessage = (messageId: string) => {
    setActiveView("chat");
    requestAnimationFrame(() => {
      document.getElementById(messageId)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  useEffect(() => {
    const close = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setPreviewImage(null);
        setHelpOpen(false);
        setGalleryOpen(false);
      }
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
      <aside className="eaa-nav" aria-label="Primary navigation">
        <div className="eaa-brand">
          <div className="eaa-brand-mark">
            <Bot size={26} aria-hidden="true" />
          </div>
          <div>
            <div className="eaa-brand-title">EAA</div>
            <div className="eaa-brand-subtitle">Agentic AI Assistant for Instrument Control</div>
          </div>
        </div>
        <nav className="eaa-nav-links">
          <button className={activeView === "chat" ? "active" : ""} type="button" onClick={() => setActiveView("chat")}>
            <MessageCircle size={22} aria-hidden="true" />
            <span>Chat</span>
          </button>
          <button className={activeView === "visualizations" ? "active" : ""} type="button" onClick={() => setActiveView("visualizations")}>
            <ChartNoAxesCombined size={22} aria-hidden="true" />
            <span>Visualizations</span>
          </button>
          <button className={activeView === "tools" ? "active" : ""} type="button" onClick={() => setActiveView("tools")}>
            <Wrench size={22} aria-hidden="true" />
            <span>Tools</span>
          </button>
          <button className={activeView === "settings" ? "active" : ""} type="button" onClick={() => setActiveView("settings")}>
            <Settings size={22} aria-hidden="true" />
            <span>Settings</span>
          </button>
        </nav>
        <div className="eaa-connection-card">
          <div className={`eaa-connection-dot eaa-connection-${connection.replace(/\W+/g, "-").toLowerCase()}`} />
          <div>
            <div>Agent Process</div>
            <span>{connection}</span>
          </div>
        </div>
      </aside>
      <div className="eaa-workspace">
        <header className="eaa-header">
          <div className="eaa-title">{uiTitle || config.title}</div>
          <div className="eaa-header-actions">
            <button className="eaa-header-button" type="button" onClick={() => setHelpOpen(true)}>
              <HelpCircle size={17} aria-hidden="true" />
              <span>Help</span>
            </button>
            <button className="eaa-icon-button" type="button" aria-label="Interrupt agent" disabled={status === "idle" && !interruptRequested} onClick={() => void requestInterrupt()}>
              <CircleStop size={18} aria-hidden="true" />
            </button>
          </div>
        </header>
        {activeView === "chat" ? (
          <main className="eaa-main" ref={mainRef} style={chatPanelStyle}>
            <section className="eaa-chat" aria-label="Chat transcript">
              <ConversationTabs
                conversations={conversations}
                activeConversationId={activeConversationId}
                onSelect={setActiveConversationId}
                onClose={closeConversation}
              />
              <div className="eaa-messages" ref={messagesRef} onScroll={registerScrollIntent}>
                {activeMessages.map((message, index) => (
                  <MessageView
                    conversationId={activeConversationId}
                    index={index}
                    key={messageKey(message, index)}
                    message={message}
                    onImage={setPreviewImage}
                    onApproval={(approved, approvalId) => submitApproval(approved, activeConversationId, approvalId)}
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
                    placeholder="Message EAA..."
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
                  <button className="eaa-attach" type="button" aria-label="Paste or attach image">
                    <Paperclip size={18} aria-hidden="true" />
                  </button>
                  <button className="eaa-send" type="submit" disabled={processing} aria-label="Send message">
                    <Send size={18} aria-hidden="true" />
                  </button>
                </div>
                <div className="eaa-disclaimer">EAA can make mistakes. Verify critical results.</div>
              </form>
            </section>
            <button
              aria-label="Resize chat and sidebar panels"
              aria-orientation="vertical"
              aria-valuemax={100}
              aria-valuemin={0}
              aria-valuenow={Math.round(chatPanelPercent)}
              className="eaa-panel-resizer eaa-panel-resizer-vertical"
              onKeyDown={onChatPanelResizeKeyDown}
              onPointerDown={startChatPanelResize}
              role="separator"
              title="Resize chat and sidebar panels"
              type="button"
            />
            <aside className="eaa-sidebar" ref={sidebarRef} aria-label="Images and logs">
              <div className="eaa-side-panel eaa-image-panel">
                <div className="eaa-panel-header">
                  <div>
                    <ImageIcon size={17} aria-hidden="true" />
                    <span>Images ({sidebarImages.length})</span>
                  </div>
                  <button className="eaa-panel-icon" type="button" aria-label="Open image gallery" onClick={() => setGalleryOpen(true)}>
                    <Maximize2 size={16} aria-hidden="true" />
                  </button>
                </div>
                <div className="eaa-images" ref={imagesRef}>
                  {sidebarImages.map((image) => (
                    <div className="eaa-sidebar-image-card" key={image.source}>
                      <button className="eaa-image-button" type="button" onClick={() => setPreviewImage(image.source)}>
                        <img className="eaa-sidebar-image" src={image.source} loading="lazy" decoding="async" alt="" />
                      </button>
                      <button className="eaa-image-title" type="button" onClick={() => jumpToImageMessage(image.messageDomId)}>
                        {image.title}
                      </button>
                    </div>
                  ))}
                  {!sidebarImages.length ? <div className="eaa-empty-inline">No images yet.</div> : null}
                </div>
              </div>
              <button
                aria-label="Resize image and log panels"
                aria-orientation="horizontal"
                aria-valuemax={100}
                aria-valuemin={0}
                aria-valuenow={Math.round(imagePanelPercent)}
                className="eaa-panel-resizer eaa-panel-resizer-horizontal"
                onKeyDown={onImagePanelResizeKeyDown}
                onPointerDown={startImagePanelResize}
                role="separator"
                title="Resize image and log panels"
                type="button"
              />
              <div className="eaa-side-panel eaa-log-panel">
                <div className="eaa-panel-header">
                  <div>
                    <TerminalSquare size={17} aria-hidden="true" />
                    <span>Live Log</span>
                  </div>
                </div>
                <div className="eaa-logs" ref={logsRef} onScroll={registerScrollIntent}>
                  {logs.map((log, index) => (
                    <div className={`eaa-log-entry eaa-log-${String(log.level || "info").toLowerCase()}`} key={logKey(log, index)}>
                      <span>{formatLogTime(log.timestamp)}</span>
                      <span>[{String(log.level || "info").toUpperCase()}]</span>
                      {log.tool_name ? <span>{log.tool_name}</span> : null}
                      <span>{log.message}</span>
                    </div>
                  ))}
                  {!logs.length ? <div className="eaa-empty-inline">No log entries yet.</div> : null}
                </div>
              </div>
            </aside>
          </main>
        ) : activeView === "visualizations" ? (
          <VisualizationsView
            conversations={conversations}
            activeConversationId={activeConversationId}
            activeConversation={activeConversation}
            onSelectConversation={setActiveConversationId}
            onCloseConversation={closeConversation}
            onImage={setPreviewImage}
          />
        ) : activeView === "tools" ? (
          <ToolsView
            tools={toolSchemas}
            reconnectingMcpServers={reconnectingMcpServers}
            mcpReconnectStatuses={mcpReconnectStatuses}
            onReconnectMcp={reconnectMcpServer}
          />
        ) : (
          <SettingsView title={uiTitle} onTitleChange={setUiTitle} />
        )}
      </div>
      {helpOpen ? <HelpDialog onClose={() => setHelpOpen(false)} /> : null}
      {galleryOpen ? (
        <ImageGalleryDialog
          images={sidebarImages}
          onClose={() => setGalleryOpen(false)}
          onPreview={setPreviewImage}
          onJumpToMessage={(messageId) => {
            setGalleryOpen(false);
            jumpToImageMessage(messageId);
          }}
        />
      ) : null}
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
