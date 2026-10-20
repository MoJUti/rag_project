import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useReadingWidth } from "./useReadingWidth";
import { ConversationTimeline } from "./ConversationTimeline";
import {
  request,
  streamChat,
  type Configuration,
  type Evidence,
  type Message,
  type Prepared,
  type Provider,
  type Session,
  type Trace,
} from "./api";

const stages: Record<string, string> = {
  history_query: "查询消歧",
  query: "查询处理",
  recall: "召回输出",
  fusion: "候选融合",
  rerank: "重排",
  evidence: "最终证据",
};
const leaf = (path: string) => path.split(/[\\/]/).pop() || path;
const seconds = (value?: number | null) =>
  value == null ? "—" : `${value.toFixed(2)}s`;
const examples = [
  "宏海科技2024年营业收入是多少？",
  "3月员工生日会的活动地点在哪里？",
  "2023年全国医院总数是多少？",
];

function ModelPicker({
  configuration,
  provider,
  disabled,
  onChange,
}: {
  configuration: Configuration | null;
  provider: Provider;
  disabled: boolean;
  onChange: (provider: Provider) => void;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const models = configuration?.providers || [];
  const modelName = (model: { id: string; model?: string }) =>
    model.model || (model.id === "zhipu" ? "GLM" : "千问");
  const current = models.find((model) => model.id === provider);

  useEffect(() => {
    function closeOutside(event: PointerEvent) {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    }
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, []);

  useEffect(() => {
    if (!open) return;
    const selected = root.current?.querySelector<HTMLButtonElement>(
      '[aria-checked="true"]:not(:disabled)',
    );
    const first = root.current?.querySelector<HTMLButtonElement>(
      '[role="menuitemradio"]:not(:disabled)',
    );
    (selected || first)?.focus();
  }, [open]);

  return (
    <div
      className="model-picker"
      ref={root}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node))
          setOpen(false);
      }}
      onKeyDown={(event) => {
        if (!open && !disabled && event.key === "ArrowDown") {
          event.preventDefault();
          setOpen(true);
          return;
        }
        if (event.key === "Escape") {
          setOpen(false);
          trigger.current?.focus();
        }
        // 菜单使用方向键导航，保留 Tab 正常离开输入区的行为。
        if (
          open &&
          ["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)
        ) {
          event.preventDefault();
          const items = Array.from(
            root.current!.querySelectorAll<HTMLButtonElement>(
              '[role="menuitemradio"]:not(:disabled)',
            ),
          );
          if (!items.length) return;
          const index = items.indexOf(
            document.activeElement as HTMLButtonElement,
          );
          const next =
            event.key === "Home"
              ? 0
              : event.key === "End"
                ? items.length - 1
                : (index +
                    (event.key === "ArrowDown" ? 1 : -1) +
                    items.length) %
                  items.length;
          items[next].focus();
        }
      }}
    >
      <button
        type="button"
        className="model-trigger"
        ref={trigger}
        aria-label="回答模型"
        aria-haspopup="menu"
        aria-expanded={open && !disabled}
        disabled={disabled}
        onClick={() => setOpen(!open)}
      >
        {current ? modelName(current) : "选择模型"}
        <svg className="model-chevron" viewBox="0 0 16 16" aria-hidden="true">
          <path d="m4 6 4 4 4-4" />
        </svg>
      </button>
      {open && !disabled && (
        <div className="model-menu" role="menu" aria-label="选择回答模型">
          {models.map((model) => (
            <button
              key={model.id}
              type="button"
              role="menuitemradio"
              aria-checked={provider === model.id}
              disabled={!model.configured}
              onClick={() => {
                onChange(model.id);
                setOpen(false);
                trigger.current?.focus();
              }}
            >
              <span>{modelName(model)}</span>
              <span aria-hidden="true">{provider === model.id ? "✓" : ""}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Inspector({
  message,
  selected,
  setSelected,
  tab,
  setTab,
}: {
  message?: Message;
  selected: string;
  setSelected: (id: string) => void;
  tab: string;
  setTab: (tab: string) => void;
}) {
  const prepared = message?.trace?.prepared || message?.prepared;
  const trace = message?.trace;
  const evidence = prepared?.evidence || [];
  const current =
    evidence.find((item) => item.citation_id === selected) || evidence[0];
  return (
    <aside className="inspector">
      <div className="inspector-title">
        <span>本次运行</span>
        <span
          className={`tag ${message?.status === "streaming" ? "live" : ""}`}
        >
          {message?.status === "streaming"
            ? "进行中"
            : message?.status === "failed"
              ? "未完成"
              : trace
                ? "已完成"
                : "等待提问"}
        </span>
      </div>
      <div className="trace-metrics">
        <div>
          <small>收到正文</small>
          <strong>{seconds(trace?.first_text_seconds)}</strong>
        </div>
        <div>
          <small>总耗时</small>
          <strong>{seconds(trace?.total_seconds)}</strong>
        </div>
        <div>
          <small>证据块</small>
          <strong>{prepared ? evidence.length : "—"}</strong>
        </div>
      </div>
      <nav className="tabs">
        {[
          ["flow", "流程"],
          ["evidence", "证据"],
          ["details", "详情"],
        ].map(([id, label]) => (
          <button
            key={id}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>
      {!prepared ? (
        <div className="panel-empty">
          <span>◈</span>
          <h3>
            {message?.status === "streaming"
              ? "正在准备检索"
              : "让每个答案都有据可查"}
          </h3>
          <p>
            提问后，这里会显示真实检索问题、阶段耗时与证据。不为旧回答重新生成流程。
          </p>
        </div>
      ) : (
        <>
          {tab === "flow" && (
            <div className="panel-content">
              <label className="section-label">实际检索问题</label>
              <div className="query-box">
                {prepared.query_resolution.query ||
                  prepared.query_resolution.clarification}
              </div>
              <p className="muted">
                消歧动作：{prepared.query_resolution.action}
              </p>
              <div className="timeline">
                {prepared.stages.map((stage, index) => (
                  <div className="stage" key={`${index}-${stage.stage}`}>
                    <i />
                    <div>
                      <b>{stages[stage.stage] || stage.stage}</b>
                      <small>
                        {stage.chunk_ids.length
                          ? `${stage.chunk_ids.length} 个切块`
                          : stage.stage === "history_query"
                            ? prepared.query_resolution.action
                            : "已执行"}
                      </small>
                    </div>
                    <span>{stage.elapsed_ms.toFixed(1)} ms</span>
                  </div>
                ))}
              </div>
              <p className="footnote">
                这里是检索管线阶段，不是整次请求的全部耗时。混合召回内部的两路候选将在后续接入。
              </p>
            </div>
          )}
          {tab === "evidence" && (
            <div className="panel-content">
              <label className="section-label">
                进入回答的证据 · {evidence.length}
              </label>
              {evidence.length === 0 ? (
                <p className="muted">
                  本次没有回答证据，可能正在澄清对象或检索没有命中。
                </p>
              ) : (
                <>
                  <div className="evidence-pills">
                    {evidence.map((item) => (
                      <button
                        key={item.citation_id}
                        onClick={() => setSelected(item.citation_id)}
                        className={
                          current?.citation_id === item.citation_id
                            ? "active"
                            : ""
                        }
                      >
                        {item.citation_id}
                      </button>
                    ))}
                  </div>
                  {current && (
                    <article className="source">
                      <h3>{leaf(current.source)}</h3>
                      <p className="source-id">{current.chunk_id}</p>
                      <pre>{current.text}</pre>
                      <details>
                        <summary>来源位置</summary>
                        <pre>{JSON.stringify(current.locations, null, 2)}</pre>
                      </details>
                    </article>
                  )}
                </>
              )}
            </div>
          )}
          {tab === "details" && (
            <div className="panel-content">
              <label className="section-label">历史上下文</label>
              <div className="detail-row">
                <span>历史消息</span>
                <b>{prepared.history.message_count} 条</b>
              </div>
              <div className="detail-row">
                <span>历史估算Token</span>
                <b>{prepared.history.estimated_tokens}</b>
              </div>
              <div className="detail-row">
                <span>包含摘要</span>
                <b>{prepared.history.has_summary ? "是" : "否"}</b>
              </div>
              <p className="footnote">不包含本次问题、系统提示或知识库证据。</p>
              <label className="section-label">实际模型调用</label>
              {trace?.model_calls?.length ? (
                trace.model_calls.map((call, i) => (
                  <div className="model-card" key={i}>
                    <b>{call.model_name || "模型名称未采集"}</b>
                    <span>
                      {call.phase === "answer" ? "回答" : "其他模型调用"} ·{" "}
                      {call.status}
                    </span>
                    <p>
                      输入{" "}
                      {call.usage?.prompt_tokens ??
                        call.usage?.input_tokens ??
                        "—"}{" "}
                      / 输出{" "}
                      {call.usage?.completion_tokens ??
                        call.usage?.output_tokens ??
                        "—"}{" "}
                      Token
                    </p>
                  </div>
                ))
              ) : (
                <p className="muted">尚未采集到模型元数据，不代表费用为零。</p>
              )}
              {prepared.query_resolution.anchors.length > 0 && (
                <>
                  <label className="section-label">消歧依据</label>
                  {prepared.query_resolution.anchors.map((anchor, i) => (
                    <blockquote key={i}>{anchor.quote}</blockquote>
                  ))}
                </>
              )}
            </div>
          )}
        </>
      )}
    </aside>
  );
}

export default function App() {
  const reading = useReadingWidth();
  const [configuration, setConfiguration] = useState<Configuration | null>(
    null,
  );
  const [sessions, setSessions] = useState<Session[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [provider, setProvider] = useState<Provider>("zhipu");
  const [busy, setBusy] = useState(false),
    [loading, setLoading] = useState(true);
  const [error, setError] = useState(""),
    [panel, setPanel] = useState(true);
  const [focused, setFocused] = useState(""),
    [selected, setSelected] = useState(""),
    [tab, setTab] = useState("flow");
  const conversation = useRef<HTMLDivElement>(null),
    busyRef = useRef(false),
    loadVersion = useRef(0);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const followBottom = useRef(true);
  const lastReply =
    messages.find((message) => message.id === focused) ||
    [...messages].reverse().find((message) => message.role === "assistant");

  async function openSession(id: string) {
    followBottom.current = true;
    const version = ++loadVersion.current;
    setActive(id);
    setLoading(true);
    // 切换时先移除上一会话，避免加载期间仍显示旧回答和旧流程。
    setMessages([]);
    setError("");
    setFocused("");
    setSelected("");
    try {
      const rows = await request<
        { role: "user" | "assistant"; content: string }[]
      >(`/api/sessions/${encodeURIComponent(id)}/messages`);
      if (version === loadVersion.current)
        setMessages(
          rows.map((row, i) => ({
            ...row,
            id: `${id}-${i}`,
            status: "complete",
          })),
        );
    } catch (exc) {
      if (version === loadVersion.current) {
        setMessages([]);
        setError(String(exc instanceof Error ? exc.message : exc));
      }
    } finally {
      if (version === loadVersion.current) setLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      request<Configuration>("/api/config"),
      request<Session[]>("/api/sessions"),
    ])
      .then(([cfg, list]) => {
        if (cancelled) return;
        setConfiguration(cfg);
        setProvider(cfg.default_provider);
        setSessions(list);
        if (list.length) void openSession(list[0].id);
        else setLoading(false);
      })
      .catch((exc) => {
        if (!cancelled) {
          setError(`无法连接后端：${exc.message}`);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    // 只滚动聊天容器，避免把整个页面和底部输入框一起带出视口。
    const container = conversation.current;
    if (container && followBottom.current)
      container.scrollTop = container.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const input = textarea.current;
    if (!input) return;
    // 随内容和阅读宽度自动调整高度，取消浏览器原生右下角拖拽角标。
    input.style.height = "auto";
    input.style.height = `${Math.max(52, Math.min(input.scrollHeight, 180, window.innerHeight / 4))}px`;
  }, [draft, reading.width]);

  async function newSession() {
    if (busyRef.current) return;
    setLoading(true);
    setError("");
    try {
      const session = await request<Session>("/api/sessions", {
        method: "POST",
      });
      setSessions((previous) => [session, ...previous]);
      setActive(session.id);
      setMessages([]);
      setFocused("");
      setSelected("");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoading(false);
    }
  }

  async function send() {
    if (busyRef.current || loading || !draft.trim() || !configuration) return;
    followBottom.current = true;
    busyRef.current = true;
    setBusy(true);
    setError("");
    const query = draft,
      replyId = crypto.randomUUID();
    setDraft("");
    setFocused(replyId);
    setTab("flow");
    setSelected("");
    const update = (patch: Partial<Message>) =>
      setMessages((previous) =>
        previous.map((message) =>
          message.id === replyId ? { ...message, ...patch } : message,
        ),
      );
    try {
      let session = active;
      if (!session) {
        const created = await request<Session>("/api/sessions", {
          method: "POST",
        });
        session = created.id;
        setActive(session);
        setSessions((previous) => [created, ...previous]);
      }
      setMessages((previous) => [
        ...previous,
        { id: crypto.randomUUID(), role: "user", content: query },
        { id: replyId, role: "assistant", content: "", status: "streaming" },
      ]);
      let content = "";
      await streamChat(session, query, provider, (event) => {
        if (event.event === "prepared")
          update({ prepared: event.data as unknown as Prepared });
        if (event.event === "token") {
          content += String(event.data.text);
          update({ content });
        }
        if (event.event === "done")
          update({
            status: "complete",
            trace: event.data.trace as unknown as Trace,
          });
        if (event.event === "error") {
          const detail = String(event.data.message);
          update({
            status: "failed",
            error: detail,
            trace: event.data.trace as unknown as Trace,
          });
          setError(detail);
        }
      });
      setSessions(await request<Session[]>("/api/sessions"));
    } catch (exc) {
      const detail = exc instanceof Error ? exc.message : String(exc);
      update({ status: "failed", error: detail });
      setError(detail);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  function showCitation(message: Message, id: string) {
    const evidence: Evidence[] =
      (message.trace?.prepared || message.prepared)?.evidence || [];
    if (!evidence.some((item) => item.citation_id === id)) {
      setError("这个回答没有对应的证据快照，不能使用其他回答的同名编号替代。");
      return;
    }
    setFocused(message.id);
    setSelected(id);
    setTab("evidence");
    setPanel(true);
  }

  return (
    <div className={`app ${panel ? "" : "panel-closed"}`}>
      <aside className="sidebar">
        <button
          className="new-chat"
          disabled={busy || loading}
          onClick={() => void newSession()}
        >
          <span>＋</span> 新建对话
        </button>
        <div className="nav-item active">
          <span>◫</span> 知识库问答
        </div>
        <div className="nav-item disabled" title="多格式文档管理将在下一批接入">
          <span>▤</span> 文档管理 <small>接入中</small>
        </div>
        <div className="section-heading">最近对话</div>
        <div className="session-list">
          {sessions.map((session) => (
            <button
              title={session.title}
              key={session.id}
              className={active === session.id ? "selected" : ""}
              disabled={busy || loading}
              onClick={() => void openSession(session.id)}
            >
              <span>◌</span>
              <span>{session.title}</span>
            </button>
          ))}
          {!sessions.length && (
            <p className="muted sidebar-empty">还没有对话，从一个问题开始。</p>
          )}
        </div>
        <div className="sidebar-bottom">
          <span className="dot" />
          <div>
            <b>本地工作区</b>
          </div>
          <span className="avatar">我</span>
        </div>
      </aside>
      <main
        className={`main ${reading.resizing ? "resizing" : ""}`}
        ref={reading.main}
        style={reading.style}
      >
        <header className="topbar">
          <div>
            <span className="breadcrumb">工作空间 /</span> 知识库问答
          </div>
          <div className="top-actions">
            <button
              className={panel ? "toggle active" : "toggle"}
              onClick={() => setPanel(!panel)}
            >
              ◈ 流程查看
            </button>
          </div>
        </header>
        <div className="conversation-shell">
          <div
            className="conversation"
            ref={conversation}
            aria-label="聊天记录"
            onScroll={(event) => {
              const area = event.currentTarget;
              followBottom.current =
                area.scrollHeight - area.scrollTop - area.clientHeight < 60;
            }}
          >
            <div className="chat-content">
              {loading && <div className="loading">正在加载工作区…</div>}
              {!loading && messages.length === 0 && (
                <section className="welcome">
                  <div className="welcome-symbol">✳</div>
                  <h1>
                    从资料出发，
                    <br />
                    让答案有迹可循。
                  </h1>
                  <p className="welcome-text">
                    向你的知识库提问，查看回答背后的证据与检索过程。
                  </p>
                  <div className="suggestions">
                    {examples.map((example) => (
                      <button key={example} onClick={() => setDraft(example)}>
                        <span>{example}</span>
                        <b>↗</b>
                      </button>
                    ))}
                  </div>
                </section>
              )}
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={`message ${message.role}`}
                  aria-label={message.role === "user" ? "你的消息" : "助手回答"}
                  data-turn-id={
                    message.role === "user" ? message.id : undefined
                  }
                >
                  <div className="message-body">
                    {message.content ? (
                      <div className="markdown">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            a: ({ href, children }) =>
                              href?.startsWith("#source-S") ? (
                                <button
                                  className="citation"
                                  onClick={() =>
                                    showCitation(message, href.slice(8))
                                  }
                                >
                                  {children}
                                </button>
                              ) : (
                                <a
                                  href={href}
                                  target="_blank"
                                  rel="noreferrer noopener"
                                >
                                  {children}
                                </a>
                              ),
                          }}
                        >
                          {message.content.replace(
                            /\[(S\d+)\](?!\()/g,
                            "[$1](#source-$1)",
                          )}
                        </ReactMarkdown>
                      </div>
                    ) : message.status === "streaming" ? (
                      <div className="thinking">
                        <i />
                        <i />
                        <i />
                        <span>正在查阅资料</span>
                      </div>
                    ) : null}
                    {message.error && (
                      <div className="message-error">{message.error}</div>
                    )}
                    {(message.trace || message.prepared) && (
                      <button
                        className="inspect-link"
                        onClick={() => {
                          setFocused(message.id);
                          setPanel(true);
                        }}
                      >
                        ◈ 查看本次流程
                      </button>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </div>
          <ConversationTimeline
            sessionId={active}
            messages={messages}
            container={conversation}
            onNavigate={() => {
              followBottom.current = false;
            }}
          />
          <div
            className="reading-handle left"
            {...reading.handleProps("left")}
          />
          <div
            className="reading-handle right"
            {...reading.handleProps("right")}
          />
        </div>
        <div className="composer-area">
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button aria-label="关闭提示" onClick={() => setError("")}>
                ×
              </button>
            </div>
          )}
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void send();
            }}
          >
            <textarea
              ref={textarea}
              aria-label="输入问题"
              placeholder="向知识库提问…"
              rows={2}
              value={draft}
              disabled={busy || loading}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <div className="composer-bottom">
              <ModelPicker
                configuration={configuration}
                provider={provider}
                disabled={busy || !configuration}
                onChange={setProvider}
              />
              <button
                className="send-button"
                aria-label="发送问题"
                aria-busy={busy}
                title={
                  busy
                    ? "正在回答"
                    : loading
                      ? "正在加载工作区"
                      : !draft.trim()
                        ? "请输入问题"
                        : "发送问题"
                }
                data-state={busy ? "busy" : "idle"}
                type="submit"
                disabled={busy || loading || !draft.trim() || !configuration}
              >
                {busy ? (
                  <svg
                    className="send-spinner"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <circle cx="12" cy="12" r="7" strokeDasharray="30 14" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 19V5m-6 6 6-6 6 6" />
                  </svg>
                )}
              </button>
            </div>
          </form>
        </div>
      </main>
      {panel && (
        <Inspector
          message={lastReply}
          selected={selected}
          setSelected={setSelected}
          tab={tab}
          setTab={setTab}
        />
      )}
    </div>
  );
}
