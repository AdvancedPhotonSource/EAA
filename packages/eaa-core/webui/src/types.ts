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
  tool_calls?: unknown;
  pending?: boolean;
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
  conversation_id?: string;
  tool_name?: string;
  arguments?: Record<string, unknown>;
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
