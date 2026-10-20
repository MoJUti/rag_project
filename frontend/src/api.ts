export type Provider = "zhipu" | "qwen";
export interface Session {
  id: string;
  title: string;
  message_count: number;
}
export interface Evidence {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  source: string;
  text: string;
  locations: Record<string, unknown>[];
}
export interface Prepared {
  query_resolution: {
    original_query: string;
    query: string;
    action: string;
    clarification: string;
    anchors: { message_index: number; quote: string }[];
  };
  stages: {
    stage: string;
    elapsed_ms: number;
    chunk_ids: string[];
    queries: string[];
  }[];
  evidence: Evidence[];
  history: {
    message_count: number;
    estimated_tokens: number;
    has_summary: boolean;
  };
}
export interface Trace {
  request_id: string;
  status: string;
  total_seconds: number;
  first_text_seconds: number | null;
  prepared: Prepared | null;
  model_calls: {
    phase: string;
    model_name?: string;
    status: string;
    usage?: Record<string, number>;
    finish_reason?: string;
  }[];
}
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "streaming" | "failed" | "complete";
  prepared?: Prepared;
  trace?: Trace;
  error?: string;
}
export interface Configuration {
  settings: Record<string, string | number>;
  default_provider: Provider;
  providers: {
    id: Provider;
    label: string;
    model: string;
    configured: boolean;
  }[];
}
export type StreamEvent = { event: string; data: Record<string, unknown> };

export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `接口请求失败（${response.status}）`,
    );
  }
  return response.json() as Promise<T>;
}

export class EventParser {
  private buffer = "";
  push(text: string): StreamEvent[] {
    this.buffer += text;
    const events: StreamEvent[] = [];
    let boundary: RegExpExecArray | null;
    // 块边界不等于事件边界；兼容任意分包和CRLF，避免丢中文或半个JSON。
    while ((boundary = /\r?\n\r?\n/.exec(this.buffer))) {
      const frame = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
      const lines = frame.split(/\r?\n/);
      const event = lines
        .find((line) => line.startsWith("event:"))
        ?.slice(6)
        .trim();
      const data = lines
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (event && data)
        events.push({
          event,
          data: JSON.parse(data) as Record<string, unknown>,
        });
    }
    return events;
  }
}

export async function streamChat(
  session: string,
  query: string,
  provider: Provider,
  receive: (event: StreamEvent) => void,
): Promise<void> {
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(session)}/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, provider }),
    },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `问答请求失败（${response.status}）`,
    );
  }
  if (!response.body) throw new Error("浏览器未收到可读取的回答流");
  const reader = response.body.getReader(),
    decoder = new TextDecoder(),
    parser = new EventParser();
  let terminal = false;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      const events = parser.push(
        done ? decoder.decode() : decoder.decode(value, { stream: true }),
      );
      for (const event of events) {
        if (event.event === "done" || event.event === "error") terminal = true;
        receive(event);
      }
      if (done) break;
    }
    if (!terminal)
      throw new Error("连接提前结束，回答尚未完整生成，请稍后重试");
  } finally {
    reader.releaseLock();
  }
}
