export type WebUIRoutes = {
  events: string;
  state: string;
  image: string;
  send: string;
  interrupt: string;
  approval: string;
  upload: string;
  skillCatalog: string;
  toolSchemas: string;
  mcpReconnect: string;
  mathjax: string;
};

export type WebUIConfig = {
  title: string;
  runtimeUrl: string;
  pollIntervalMs: number;
  routes: WebUIRoutes;
};

export type WebUIMessage = {
  id?: string | number;
  role?: string;
  content?: string;
  image?: string;
  images?: string[];
  timestamp?: string;
  tool_calls?: unknown;
  pending?: boolean;
  approval_id?: string;
  approval_requested_at?: string;
  approval_expires_at?: string;
  approval_timeout_seconds?: number;
};

export type RuntimeLogEntry = {
  id: string;
  timestamp: string;
  source: string;
  level: string;
  message: string;
  tool_name?: string | null;
  progress?: number | null;
  total?: number | null;
};

export type RuntimeSnapshot = {
  conversations?: RuntimeConversation[];
  messages?: WebUIMessage[];
  logs?: RuntimeLogEntry[];
  status?: string;
  input_requested?: boolean;
  interrupt_requested?: boolean;
  pending_approval?: PendingApproval | null;
};

export type PendingApproval = {
  id?: string;
  conversation_id?: string;
  tool_name?: string;
  arguments?: Record<string, unknown>;
  requested_at?: string;
  expires_at?: string;
  timeout_seconds?: number;
};

export type RuntimeConversation = {
  id: string;
  label: string;
  kind: "primary" | "subagent" | string;
  status?: string;
  terminated?: boolean;
  messages?: WebUIMessage[];
  pending_approval?: PendingApproval | null;
};

export type Skill = {
  name?: string;
  description?: string;
};

export type ToolSchema = {
  type?: string;
  mcp?: {
    server_id?: string;
    server_name?: string;
    tool_name?: string;
  };
  function?: {
    name?: string;
    description?: string;
    parameters?: {
      properties?: Record<string, unknown>;
      required?: string[];
      [key: string]: unknown;
    };
  };
};

declare global {
  interface Window {
    EAA_WEBUI_CONFIG?: WebUIConfig;
    MathJax?: {
      tex?: Record<string, unknown>;
      svg?: Record<string, unknown>;
      typesetPromise?: (elements: Element[]) => Promise<void>;
    };
  }
}
