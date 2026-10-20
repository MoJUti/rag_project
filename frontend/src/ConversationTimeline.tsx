import { useEffect, useId, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "./api";

const TICK_HEIGHT = 10;
const HOVER_PROGRESS = [1, 0.7, 0.4, 0.2];

export function markerWidth(distance: number): number {
  return 6 + 20 * (HOVER_PROGRESS[distance] || 0);
}

export function edgeFade(top: number, total: number, height: number) {
  return {
    top: Math.min(40, Math.max(0, top)),
    bottom: Math.min(40, Math.max(0, total - height - top)),
  };
}

export function ConversationTimeline({
  messages,
  container,
  onNavigate,
  sessionId,
}: {
  messages: Message[];
  container: RefObject<HTMLDivElement | null>;
  onNavigate: () => void;
  sessionId?: string | null;
}) {
  const turns = messages.flatMap((message, index) =>
    message.role === "user"
      ? [
          {
            ...message,
            answer:
              messages[index + 1]?.role === "assistant"
                ? messages[index + 1].content
                : "",
          },
        ]
      : [],
  );
  const enabled = turns.length >= 4;
  const [active, setActive] = useState("");
  const [hovered, setHovered] = useState<number | null>(null);
  const [preview, setPreview] = useState<{ id: string; top: number } | null>(
    null,
  );
  const [edges, setEdges] = useState({ top: 0, bottom: 0 });
  const [bookmarks, setBookmarks] = useState<number[]>([]);
  const [storageError, setStorageError] = useState("");
  const rail = useRef<HTMLDivElement>(null);
  const card = useRef<HTMLDivElement>(null);
  const openTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const previewId = useId();
  // 收藏只存会话ID与轮次，不存正文，也不依赖重新加载后会变化的临时消息ID。
  const bookmarkKey = sessionId ? `rag.timeline.bookmarks:${sessionId}` : null;

  useEffect(() => {
    setPreview(null);
    setHovered(null);
    setBookmarks([]);
    setStorageError("");
    clearTimeout(openTimer.current);
    clearTimeout(closeTimer.current);
    if (!bookmarkKey) return;
    try {
      const saved: unknown = JSON.parse(
        localStorage.getItem(bookmarkKey) || "[]",
      );
      if (
        !Array.isArray(saved) ||
        !saved.every((item) => Number.isInteger(item) && item >= 0)
      )
        throw new Error("收藏记录格式不正确");
      setBookmarks(saved);
    } catch {
      setStorageError("无法读取本机收藏，导航仍可使用。");
    }
  }, [bookmarkKey, turns[0]?.id]);

  useEffect(
    () => () => {
      clearTimeout(openTimer.current);
      clearTimeout(closeTimer.current);
    },
    [],
  );

  function updateEdges() {
    const list = rail.current;
    if (list)
      setEdges(edgeFade(list.scrollTop, list.scrollHeight, list.clientHeight));
  }

  useEffect(() => {
    if (!enabled) return;
    const element = container.current;
    if (!element) return;
    function update() {
      const line = element!.getBoundingClientRect().top + 24;
      const nodes = Array.from(
        element!.querySelectorAll<HTMLElement>("[data-turn-id]"),
      );
      let current = nodes[0];
      for (const node of nodes) {
        if (node.getBoundingClientRect().top > line) break;
        current = node;
      }
      setActive(current?.dataset.turnId || "");
      updateEdges();
    }
    update();
    element.addEventListener("scroll", update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(element);
    if (rail.current) observer.observe(rail.current);
    const content = element.querySelector(".chat-content");
    if (content) observer.observe(content);
    return () => {
      element.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, [container, enabled, turns.length, turns[0]?.id]);

  useEffect(() => {
    const list = rail.current;
    if (!list || hovered !== null) return;
    const index = turns.findIndex((turn) => turn.id === active);
    if (index < 0) return;
    const top = index * TICK_HEIGHT;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (top + TICK_HEIGHT > list.scrollTop + list.clientHeight)
      list.scrollTop = top + TICK_HEIGHT - list.clientHeight;
    updateEdges();
  }, [active, enabled]);

  function closePreview() {
    clearTimeout(openTimer.current);
    clearTimeout(closeTimer.current);
    setHovered(null);
    setPreview(null);
  }

  function leave() {
    clearTimeout(openTimer.current);
    clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(closePreview, 180);
  }

  function show(index: number, button: HTMLElement, immediate = false) {
    clearTimeout(closeTimer.current);
    clearTimeout(openTimer.current);
    setHovered(index);
    const shell = container.current?.parentElement;
    const top =
      button.getBoundingClientRect().top +
      button.offsetHeight / 2 -
      (shell?.getBoundingClientRect().top || 0);
    const next = { id: turns[index].id, top };
    if (immediate || preview) setPreview(next);
    else openTimer.current = setTimeout(() => setPreview(next), 150);
  }

  function jump(id: string) {
    const element = container.current;
    const target = Array.from(
      element?.querySelectorAll<HTMLElement>("[data-turn-id]") || [],
    ).find((node) => node.dataset.turnId === id);
    if (!element || !target) return;
    onNavigate();
    const top = Math.max(
      0,
      element.scrollTop +
        target.getBoundingClientRect().top -
        element.getBoundingClientRect().top -
        20,
    );
    element.scrollTo({
      top,
      behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
    });
    setActive(id);
    closePreview();
  }

  function toggleBookmark(index: number) {
    const next = bookmarks.includes(index)
      ? bookmarks.filter((item) => item !== index)
      : [...bookmarks, index];
    try {
      if (bookmarkKey) localStorage.setItem(bookmarkKey, JSON.stringify(next));
      setBookmarks(next);
      setStorageError("");
    } catch {
      setStorageError("收藏保存失败，请检查浏览器的本地存储权限。");
    }
  }

  const questionIndex = turns.findIndex((turn) => turn.id === preview?.id);
  const question = turns[questionIndex];
  if (!enabled) return null;

  return (
    <>
      <nav
        className="conversation-timeline"
        aria-label="对话时间轴"
        style={{ "--turn-count": turns.length } as CSSProperties}
        onMouseEnter={() => clearTimeout(closeTimer.current)}
        onMouseLeave={leave}
        onBlur={(event) => {
          if (
            !event.currentTarget.contains(event.relatedTarget as Node) &&
            !card.current?.contains(event.relatedTarget as Node)
          )
            leave();
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            closePreview();
            return;
          }
          if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key))
            return;
          event.preventDefault();
          const buttons = Array.from(
            rail.current!.querySelectorAll<HTMLButtonElement>("button"),
          );
          const current = buttons.indexOf(
            document.activeElement as HTMLButtonElement,
          );
          const next =
            event.key === "Home"
              ? 0
              : event.key === "End"
                ? buttons.length - 1
                : (current +
                    (event.key === "ArrowDown" ? 1 : -1) +
                    buttons.length) %
                  buttons.length;
          const list = rail.current!;
          if (next * TICK_HEIGHT < list.scrollTop)
            list.scrollTop = next * TICK_HEIGHT;
          else if (
            (next + 1) * TICK_HEIGHT >
            list.scrollTop + list.clientHeight
          )
            list.scrollTop = (next + 1) * TICK_HEIGHT - list.clientHeight;
          updateEdges();
          buttons[next]?.focus({ preventScroll: true });
        }}
      >
        <div
          className="timeline-rail"
          ref={rail}
          style={
            {
              "--fade-top": `${edges.top}px`,
              "--fade-bottom": `${edges.bottom}px`,
            } as CSSProperties
          }
          onScroll={() => {
            updateEdges();
            closePreview();
          }}
        >
          {turns.map((turn, index) => (
            <button
              key={turn.id}
              type="button"
              aria-label={`跳转到第${index + 1}轮：${turn.content}`}
              aria-current={active === turn.id ? "location" : undefined}
              aria-describedby={preview?.id === turn.id ? previewId : undefined}
              data-bookmarked={bookmarks.includes(index) || undefined}
              data-hovered={hovered === index || undefined}
              style={
                {
                  "--tick-scale":
                    markerWidth(
                      hovered === null ? 99 : Math.abs(index - hovered),
                    ) / 26,
                } as CSSProperties
              }
              onClick={() => jump(turn.id)}
              onMouseEnter={(event) => show(index, event.currentTarget)}
              onFocus={(event) => show(index, event.currentTarget, true)}
            >
              <span className="timeline-marker">
                <span className="timeline-marker-line" />
                {bookmarks.includes(index) && <i />}
              </span>
            </button>
          ))}
        </div>
      </nav>
      {question && preview && (
        <div
          id={previewId}
          role="region"
          aria-label="对话预览"
          ref={card}
          className="turn-preview"
          style={{ top: `clamp(80px, ${preview.top}px, calc(100% - 80px))` }}
          onMouseEnter={() => clearTimeout(closeTimer.current)}
          onMouseLeave={leave}
          onKeyDown={(event) => {
            if (event.key === "Escape") closePreview();
          }}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node))
              leave();
          }}
        >
          <div className="turn-preview-heading">
            <strong>{question.content.replace(/\s+/g, " ")}</strong>
            <button
              type="button"
              aria-label={
                bookmarks.includes(questionIndex)
                  ? "取消收藏此轮对话"
                  : "收藏此轮对话"
              }
              aria-pressed={bookmarks.includes(questionIndex)}
              onClick={() => toggleBookmark(questionIndex)}
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M4 2.5h8v11L8 11l-4 2.5z" />
              </svg>
            </button>
          </div>
          <div className="preview-response">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                img: ({ alt }) => <span>{alt || "图片"}</span>,
                a: ({ children }) => <span>{children}</span>,
              }}
            >
              {question.answer.slice(0, 2000) || "暂无回答"}
            </ReactMarkdown>
          </div>
          {storageError && (
            <div role="alert" className="bookmark-error">
              {storageError}
            </div>
          )}
        </div>
      )}
    </>
  );
}
