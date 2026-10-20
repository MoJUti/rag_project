// @vitest-environment jsdom
import { useRef } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import {
  ConversationTimeline,
  edgeFade,
  markerWidth,
} from "./ConversationTimeline";
import { act } from "@testing-library/react";
import { readFileSync } from "node:fs";
import type { Message } from "./api";

const messages: Message[] = [
  { id: "one", role: "user", content: "第一个问题" },
  { id: "reply-one", role: "assistant", content: "第一个回答" },
  { id: "two", role: "user", content: "第二个问题" },
  { id: "reply-two", role: "assistant", content: "第二个回答" },
  { id: "three", role: "user", content: "第三个问题" },
  { id: "reply-three", role: "assistant", content: "第三个回答" },
  { id: "four", role: "user", content: "第四个问题" },
  { id: "reply-four", role: "assistant", content: "第四个回答" },
];
const navigate = vi.fn();

function Harness({ rows = messages }: { rows?: Message[] }) {
  const container = useRef<HTMLDivElement>(null);
  return (
    <>
      <div ref={container} aria-label="测试聊天区">
        <div className="chat-content">
          {rows
            .filter((row) => row.role === "user")
            .map((row) => (
              <article key={row.id} data-turn-id={row.id}>
                {row.content}
              </article>
            ))}
        </div>
      </div>
      <ConversationTimeline
        sessionId="test_timeline"
        messages={rows}
        container={container}
        onNavigate={navigate}
      />
    </>
  );
}

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  navigate.mockClear();
});

it("每轮一个刻度，悬停预览本轮问题及回答，离开后收起", () => {
  render(<Harness />);
  const nav = screen.getByRole("navigation", { name: "对话时间轴" });
  expect(within(nav).getAllByRole("button")).toHaveLength(4);
  fireEvent.mouseEnter(
    screen.getByRole("button", { name: "跳转到第2轮：第二个问题" }),
  );
  expect(screen.queryByRole("region", { name: "对话预览" })).toBeNull();
  act(() => vi.advanceTimersByTime(150));
  const preview = screen.getByRole("region", { name: "对话预览" });
  expect(within(preview).getByText("第二个问题")).toBeTruthy();
  expect(within(preview).getByText("第二个回答")).toBeTruthy();
  expect(within(preview).queryByText("第一个回答")).toBeNull();
  fireEvent.mouseLeave(nav);
  act(() => vi.advanceTimersByTime(180));
  expect(screen.queryByRole("region", { name: "对话预览" })).toBeNull();
});

it("点击按目标几何位置滚动聊天区，不滚动整个页面", () => {
  render(<Harness />);
  const container = screen.getByLabelText("测试聊天区");
  const target = container.querySelector<HTMLElement>('[data-turn-id="two"]')!;
  container.scrollTop = 200;
  vi.spyOn(container, "getBoundingClientRect").mockReturnValue({
    top: 100,
  } as DOMRect);
  vi.spyOn(target, "getBoundingClientRect").mockReturnValue({
    top: 500,
  } as DOMRect);
  const scroll = vi.fn();
  container.scrollTo = scroll;
  const pageScroll = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  fireEvent.click(
    screen.getByRole("button", { name: "跳转到第2轮：第二个问题" }),
  );
  expect(navigate).toHaveBeenCalledOnce();
  expect(scroll).toHaveBeenCalledWith({ top: 580, behavior: "smooth" });
  expect(pageScroll).not.toHaveBeenCalled();
  expect(
    screen
      .getByRole("button", { name: "跳转到第2轮：第二个问题" })
      .getAttribute("aria-current"),
  ).toBe("location");
});

it("方向键移动刻度焦点，Esc收起预览", () => {
  render(<Harness />);
  const first = screen.getByRole("button", { name: "跳转到第1轮：第一个问题" });
  fireEvent.focus(first);
  expect(screen.getByRole("region", { name: "对话预览" })).toBeTruthy();
  act(() => first.focus());
  fireEvent.keyDown(first, { key: "ArrowDown" });
  expect(document.activeElement).toBe(
    screen.getByRole("button", { name: "跳转到第2轮：第二个问题" }),
  );
  fireEvent.keyDown(document.activeElement!, { key: "Escape" });
  expect(screen.queryByRole("region", { name: "对话预览" })).toBeNull();
});

it("空会话不显示时间轴，未回答的轮次不串用其他回答", () => {
  const view = render(<Harness rows={[]} />);
  expect(screen.queryByRole("navigation")).toBeNull();
  view.rerender(
    <Harness
      rows={[messages[0], messages[2], messages[3], messages[4], messages[6]]}
    />,
  );
  fireEvent.focus(
    screen.getByRole("button", { name: "跳转到第1轮：第一个问题" }),
  );
  expect(
    within(screen.getByRole("region", { name: "对话预览" })).getByText(
      "暂无回答",
    ),
  ).toBeTruthy();
  expect(
    within(screen.getByRole("region", { name: "对话预览" })).queryByText(
      "第二个回答",
    ),
  ).toBeNull();
});

it("不足四条用户消息隐藏，第四条到来才显示，刻度数量不包含回答", () => {
  const view = render(<Harness rows={messages.slice(0, 6)} />);
  expect(screen.queryByRole("navigation")).toBeNull();
  view.rerender(<Harness />);
  const nav = screen.getByRole("navigation");
  expect(nav.style.getPropertyValue("--turn-count")).toBe("4");
  expect(within(nav).getAllByRole("button")).toHaveLength(4);
});

it("悬停项与附近三档逐级变长，离开后恢复短刻度", () => {
  expect([0, 1, 2, 3, 4, 10].map(markerWidth)).toEqual([26, 20, 14, 10, 6, 6]);
  render(<Harness />);
  const nav = screen.getByRole("navigation");
  const ticks = within(nav).getAllByRole("button");
  fireEvent.mouseEnter(ticks[0]);
  for (const [index, tick] of ticks.entries()) {
    expect(Number(tick.style.getPropertyValue("--tick-scale"))).toBeCloseTo(
      markerWidth(index) / 26,
    );
  }
  fireEvent.mouseLeave(nav);
  act(() => vi.advanceTimersByTime(180));
  expect(Number(ticks[0].style.getPropertyValue("--tick-scale"))).toBeCloseTo(
    6 / 26,
  );
});

it("溢出后刻度栏独立滚动，渐隐只在仍有隐藏内容的一端出现", () => {
  expect(edgeFade(0, 100, 200)).toEqual({ top: 0, bottom: 0 });
  render(<Harness />);
  const list = screen
    .getByRole("navigation")
    .querySelector<HTMLElement>(".timeline-rail")!;
  Object.defineProperties(list, {
    scrollHeight: { value: 1000 },
    clientHeight: { value: 400 },
  });
  list.scrollTop = 0;
  fireEvent.scroll(list);
  expect(list.style.getPropertyValue("--fade-top")).toBe("0px");
  expect(list.style.getPropertyValue("--fade-bottom")).toBe("40px");
  list.scrollTop = 300;
  fireEvent.scroll(list);
  expect(list.style.getPropertyValue("--fade-top")).toBe("40px");
  expect(list.style.getPropertyValue("--fade-bottom")).toBe("40px");
  list.scrollTop = 600;
  fireEvent.scroll(list);
  expect(list.style.getPropertyValue("--fade-bottom")).toBe("0px");
  expect(screen.getByLabelText("测试聊天区").scrollTop).toBe(0);
  expect(navigate).not.toHaveBeenCalled();
});

it("固定间距、垂直居中和高度上限写入真实样式，不再生成装饰性假刻度", () => {
  const style = document.createElement("style");
  style.textContent = readFileSync("src/styles.css", "utf8");
  document.head.append(style);
  const rules = Array.from(style.sheet!.cssRules) as CSSStyleRule[];
  const nav = rules.find(
    (rule) => rule.selectorText === ".conversation-timeline",
  )!;
  expect(nav.style.top).toBe("50%");
  expect(nav.style.height).toContain("--turn-count");
  expect(nav.style.height).toContain("70vh");
  expect(nav.style.height).toContain("40rem");
  expect(
    rules.find((rule) => rule.selectorText === ".timeline-rail button")!.style
      .height,
  ).toBe("10px");
  const list = rules.find((rule) => rule.selectorText === ".timeline-rail")!;
  expect(list.style.getPropertyValue("mask-image")).toContain(
    "linear-gradient",
  );
  expect(list.style.background).not.toContain("repeating-linear-gradient");
  style.remove();
});

it("收藏可以取消、刷新后恢复且不保存正文，存储失败明确报错", () => {
  const view = render(<Harness />);
  fireEvent.focus(
    screen.getByRole("button", { name: "跳转到第1轮：第一个问题" }),
  );
  fireEvent.click(screen.getByRole("button", { name: "收藏此轮对话" }));
  expect(localStorage.getItem("rag.timeline.bookmarks:test_timeline")).toBe(
    "[0]",
  );
  expect(
    screen
      .getByRole("button", { name: "取消收藏此轮对话" })
      .getAttribute("aria-pressed"),
  ).toBe("true");
  view.unmount();
  render(<Harness />);
  expect(
    screen.getByRole("button", { name: "跳转到第1轮：第一个问题" }).dataset
      .bookmarked,
  ).toBe("true");
  fireEvent.focus(
    screen.getByRole("button", { name: "跳转到第1轮：第一个问题" }),
  );
  fireEvent.click(screen.getByRole("button", { name: "取消收藏此轮对话" }));
  expect(localStorage.getItem("rag.timeline.bookmarks:test_timeline")).toBe(
    "[]",
  );
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("存储不可用");
  });
  fireEvent.click(screen.getByRole("button", { name: "收藏此轮对话" }));
  expect(screen.getByRole("alert").textContent).toContain("收藏保存失败");
});

it("切换历史后取消待显示的旧预览，不串用旧问题", () => {
  const view = render(<Harness />);
  fireEvent.mouseEnter(
    screen.getByRole("button", { name: "跳转到第1轮：第一个问题" }),
  );
  view.rerender(
    <Harness
      rows={messages.map((message) => ({
        ...message,
        id: `new-${message.id}`,
        content: `新会话${message.content}`,
      }))}
    />,
  );
  act(() => vi.advanceTimersByTime(300));
  expect(screen.queryByRole("region", { name: "对话预览" })).toBeNull();
});
