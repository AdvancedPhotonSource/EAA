export type WebUIRoutes = {
  events: string;
  state: string;
  image: string;
  send: string;
  interrupt: string;
  approval: string;
  upload: string;
  skillCatalog: string;
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

export type RuntimeSnapshot = {
  messages?: WebUIMessage[];
  status?: string;
  input_requested?: boolean;
  interrupt_requested?: boolean;
  pending_approval?: PendingApproval | null;
};

export type PendingApproval = {
  tool_name?: string;
  arguments?: Record<string, unknown>;
};

export type Skill = {
  name?: string;
  description?: string;
};

declare global {
  interface Window {
    EAA_WEBUI_CONFIG?: WebUIConfig;
    MathJax?: {
      typesetPromise?: (elements: Element[]) => Promise<void>;
    };
  }
}
