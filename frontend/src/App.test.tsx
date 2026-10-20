// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  cleanup,
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import App from "./App";
import { readFileSync } from "node:fs";

beforeEach(() => {
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
});

it("双侧拖拽保持正文和输入框同宽，并限制上下限和窗口可用宽度", async () => {
  let available = 1600;
  let notifyResize = () => {};
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(
    () => available,
  );
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(callback: () => void) {
        notifyResize = callback;
      }
      observe() {}
      disconnect() {}
    },
  );
  // JSDOM 没有完整的 PointerEvent，补齐本次交互实际使用的指针编号。
  vi.stubGlobal(
    "PointerEvent",
    class extends MouseEvent {
      pointerId: number;
      constructor(type: string, init: PointerEventInit = {}) {
        super(type, init);
        this.pointerId = init.pointerId ?? 1;
      }
    },
  );
  vi.stubGlobal("fetch", async (url: string) => {
    if (url === "/api/config")
      return Response.json({
        default_provider: "zhipu",
        settings: {},
        providers: [],
      });
    if (url === "/api/sessions") return Response.json([]);
    throw new Error(`未预期请求：${url}`);
  });
  const style = document.createElement("style");
  style.textContent = readFileSync("src/styles.css", "utf8");
  document.head.append(style);
  render(<App />);
  await screen.findByText("从资料出发，", { exact: false });
  const left = screen.getByRole("separator", { name: "左侧调整对话宽度" });
  const right = screen.getByRole("separator", { name: "右侧调整对话宽度" });
  left.setPointerCapture = vi.fn();
  right.setPointerCapture = vi.fn();
  const width = () => Number(right.getAttribute("aria-valuenow"));
  const drag = (handle: HTMLElement, start: number, end: number) => {
    fireEvent.pointerDown(handle, { button: 0, pointerId: 1, clientX: start });
    fireEvent.pointerMove(handle, { pointerId: 1, clientX: end });
    fireEvent.pointerUp(handle, { pointerId: 1 });
  };
  expect(width()).toBe(960);
  drag(right, 100, 150);
  expect(width()).toBe(1060);
  drag(left, 100, 150);
  expect(width()).toBe(960);
  drag(left, 0, 5000);
  expect(width()).toBe(560);
  drag(right, 0, 5000);
  expect(width()).toBe(1280);
  fireEvent.pointerMove(right, { pointerId: 1, clientX: 0 });
  expect(width()).toBe(1280);
  fireEvent.doubleClick(right);
  expect(width()).toBe(960);
  fireEvent.keyDown(left, { key: "Home" });
  expect(width()).toBe(560);
  fireEvent.keyDown(right, { key: "End" });
  expect(width()).toBe(1280);
  available = 760;
  act(() => notifyResize());
  expect(width()).toBe(568);
  const main = right.closest("main")!;
  expect(main.style.getPropertyValue("--reading-width")).toBe("568px");
  expect(getComputedStyle(main.querySelector(".chat-content")!).width).toBe(
    "var(--reading-width)",
  );
  expect(getComputedStyle(main.querySelector(".composer")!).width).toBe(
    "min(100%, var(--reading-width))",
  );
  expect(
    getComputedStyle(screen.getByRole("textbox", { name: "输入问题" })).resize,
  ).toBe("none");
  expect(screen.getByText("最近对话").textContent).toBe("最近对话");
  // 检查真实伪元素样式，避免拖拽线再次退化为紧贴正文的实色线。
  const rules = Array.from(style.sheet!.cssRules) as CSSStyleRule[];
  const line = rules.find(
    (rule) => rule.selectorText === ".reading-handle::after",
  )!;
  expect(line.style.background).toContain("linear-gradient");
  expect(line.style.background).toContain("transparent");
  for (const side of ["left", "right"]) {
    const handle = rules.find(
      (rule) => rule.selectorText === `.reading-handle.${side}`,
    )!;
    expect(handle.style.getPropertyValue(side)).toContain("36px");
  }
  style.remove();
});

it("输入框内选择模型，只展示名称，支持关闭菜单且按所选模型发送", async () => {
  let payload: unknown;
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    if (url === "/api/config")
      return Response.json({
        default_provider: "zhipu",
        settings: {},
        providers: [
          {
            id: "zhipu",
            label: "GLM · low",
            model: "glm-5.3-flash",
            configured: true,
          },
          {
            id: "qwen",
            label: "千问 · 关闭思考",
            model: "qwen3.8-max",
            configured: true,
          },
        ],
      });
    if (url === "/api/sessions" && init?.method === "POST")
      return Response.json({ id: "picker", title: "新对话", message_count: 0 });
    if (url === "/api/sessions") return Response.json([]);
    if (url.endsWith("/chat")) {
      payload = JSON.parse(String(init?.body));
      return new Response(
        'event: token\ndata: {"text":"切换成功"}\n\nevent: done\ndata: {}\n\n',
      );
    }
    throw new Error(`未预期请求：${url}`);
  });
  render(<App />);
  const trigger = screen.getByRole("button", { name: "回答模型" });
  await waitFor(() => expect(trigger.hasAttribute("disabled")).toBe(false));
  expect(trigger.closest(".composer")).not.toBeNull();
  expect(trigger.textContent).toContain("glm-5.3-flash");
  expect(
    screen.queryByText(/知序|关闭思考|low|资料与密钥由后端管理/),
  ).toBeNull();
  fireEvent.click(trigger);
  fireEvent.click(screen.getByRole("menuitemradio", { name: "qwen3.8-max" }));
  expect(screen.queryByRole("menu")).toBeNull();
  expect(trigger.textContent).toContain("qwen3.8-max");
  fireEvent.click(trigger);
  expect(document.activeElement).toBe(
    screen.getByRole("menuitemradio", { name: "qwen3.8-max" }),
  );
  fireEvent.keyDown(document.activeElement!, { key: "Escape" });
  expect(screen.queryByRole("menu")).toBeNull();
  fireEvent.click(trigger);
  fireEvent.pointerDown(document.body);
  expect(screen.queryByRole("menu")).toBeNull();
  fireEvent.change(screen.getByRole("textbox", { name: "输入问题" }), {
    target: { value: "测试切换" },
  });
  fireEvent.click(screen.getByRole("button", { name: "发送问题" }));
  await screen.findByText("切换成功");
  expect(payload).toEqual({ query: "测试切换", provider: "qwen" });
});

it("回看历史时新正文不抢回底部，回到底部后恢复跟随", async () => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(value) {
      controller = value;
    },
  });
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    if (url === "/api/config")
      return Response.json({
        default_provider: "zhipu",
        settings: {},
        providers: [
          { id: "zhipu", model: "glm-test", label: "GLM", configured: true },
        ],
      });
    if (url === "/api/sessions" && init?.method === "POST")
      return Response.json({ id: "follow", title: "新对话", message_count: 0 });
    if (url === "/api/sessions") return Response.json([]);
    if (url.endsWith("/chat")) return new Response(body);
    throw new Error(`未预期请求：${url}`);
  });
  render(<App />);
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "回答模型" }).hasAttribute("disabled"),
    ).toBe(false),
  );
  fireEvent.change(screen.getByRole("textbox", { name: "输入问题" }), {
    target: { value: "持续回答" },
  });
  const sendButton = screen.getByRole("button", { name: "发送问题" });
  expect(sendButton.hasAttribute("disabled")).toBe(false);
  expect(sendButton.querySelector("svg path")).not.toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "发送问题" }));
  const emit = (text: string) =>
    controller.enqueue(
      new TextEncoder().encode(
        `event: token\ndata: ${JSON.stringify({ text })}\n\n`,
      ),
    );
  emit("第一段");
  await screen.findByText("第一段");
  expect(sendButton.getAttribute("aria-busy")).toBe("true");
  expect(sendButton.hasAttribute("disabled")).toBe(true);
  expect(sendButton.title).toBe("正在回答");
  expect(sendButton.querySelector(".send-spinner circle")).not.toBeNull();
  const area = screen.getByLabelText("聊天记录");
  Object.defineProperties(area, {
    scrollHeight: { value: 1000 },
    clientHeight: { value: 400 },
  });
  area.scrollTop = 150;
  fireEvent.scroll(area);
  emit("第二段");
  await screen.findByText("第一段第二段");
  expect(area.scrollTop).toBe(150);
  area.scrollTop = 600;
  fireEvent.scroll(area);
  emit("第三段");
  await screen.findByText("第一段第二段第三段");
  expect(area.scrollTop).toBe(1000);
  controller.enqueue(new TextEncoder().encode("event: done\ndata: {}\n\n"));
  controller.close();
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "回答模型" }).hasAttribute("disabled"),
    ).toBe(false),
  );
  expect(sendButton.getAttribute("aria-busy")).toBe("false");
  expect(sendButton.querySelector(".send-spinner")).toBeNull();
  expect(sendButton.hasAttribute("disabled")).toBe(true);
  expect(sendButton.title).toBe("请输入问题");
});

it("模型初始化失败时明确报错，不残留生成中的动画", async () => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    if (url === "/api/config")
      return Response.json({
        default_provider: "zhipu",
        settings: {},
        providers: [
          {
            id: "zhipu",
            label: "GLM · low",
            model: "glm-test",
            configured: true,
          },
        ],
      });
    if (url === "/api/sessions" && init?.method === "POST")
      return Response.json({
        id: "chat_test",
        title: "新对话",
        message_count: 0,
      });
    if (url === "/api/sessions") return Response.json([]);
    return new Response(
      'event: error\ndata: {"message":"模型连接失败","trace":{"request_id":"failure","status":"failed","prepared":null,"model_calls":[],"first_text_seconds":null,"total_seconds":0.2}}\n\n',
    );
  });
  render(<App />);
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "回答模型" }).hasAttribute("disabled"),
    ).toBe(false),
  );
  fireEvent.change(screen.getByRole("textbox", { name: "输入问题" }), {
    target: { value: "测试问题" },
  });
  fireEvent.click(screen.getByRole("button", { name: "发送问题" }));
  await waitFor(() =>
    expect(screen.getAllByText("模型连接失败").length).toBeGreaterThan(0),
  );
  expect(screen.queryByText("正在查阅资料")).toBeNull();
  expect(
    screen.getByRole("button", { name: "发送问题" }).getAttribute("aria-busy"),
  ).toBe("false");
});

it("发送按钮为等宽高圆形图标按钮，禁用和加载状态有独立样式", () => {
  const style = document.createElement("style");
  style.textContent = readFileSync("src/styles.css", "utf8");
  document.head.append(style);
  const rules = Array.from(style.sheet!.cssRules) as CSSStyleRule[];
  const button = rules.find(
    (rule) => rule.selectorText === ".composer-bottom > .send-button",
  )!;
  expect(button.style.borderRadius).toBe("50%");
  expect(button.style.width).toBe(button.style.height);
  expect(button.style.width).toBe("36px");
  expect(
    rules.find((rule) => rule.selectorText === ".send-spinner")!.style
      .animation,
  ).toContain("send-spin");
  expect(
    rules.find(
      (rule) =>
        rule.selectorText === ".composer-bottom > .send-button:disabled",
    )!.style.opacity,
  ).toBe("1");
  style.remove();
});

it("加载配置默认GLM，发送一次问题并呈现流式回答与耗时", async () => {
  const scrollIntoView = vi.fn();
  Element.prototype.scrollIntoView = scrollIntoView;
  const queries: unknown[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url === "/api/config")
      return Response.json({
        default_provider: "zhipu",
        settings: { hybrid_final_k: 10 },
        providers: [
          {
            id: "zhipu",
            label: "GLM · low",
            model: "glm-test",
            configured: true,
          },
        ],
      });
    if (url === "/api/sessions" && init?.method === "POST")
      return Response.json({
        id: "chat_test",
        title: "新对话",
        message_count: 0,
      });
    if (url === "/api/sessions") return Response.json([]);
    if (url.endsWith("/chat")) {
      queries.push(JSON.parse(String(init?.body)));
      return new Response(
        'event: token\ndata: {"text":"答案是47。"}\n\nevent: done\ndata: {"trace":{"request_id":"test","status":"completed","total_seconds":1.2,"first_text_seconds":0.5,"prepared":null,"model_calls":[]}}\n\n',
      );
    }
    throw new Error(`未预期请求：${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<App />);
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "回答模型" }).getAttribute("disabled"),
    ).toBeNull(),
  );
  fireEvent.change(screen.getByRole("textbox", { name: "输入问题" }), {
    target: { value: "费用是多少？" },
  });
  fireEvent.click(screen.getByRole("button", { name: "发送问题" }));
  await screen.findByText("答案是47。");
  expect(queries).toEqual([{ query: "费用是多少？", provider: "zhipu" }]);
  expect(screen.getByText("1.20s")).toBeTruthy();
  const user = screen.getByRole("article", { name: "你的消息" });
  const assistant = screen.getByRole("article", { name: "助手回答" });
  expect(user.classList.contains("user")).toBe(true);
  expect(user.querySelector(".message-avatar")).toBeNull();
  expect(assistant.classList.contains("assistant")).toBe(true);
  expect(assistant.querySelector(".message-avatar")).toBeNull();
  // 消息更新不能再滚动整个页面；只允许聊天容器内部滚动。
  expect(scrollIntoView).not.toHaveBeenCalled();
  expect(screen.getByRole("textbox", { name: "输入问题" })).toBeTruthy();
});

it("长聊天使用独立滚动容器，网格和输入区保留高度约束", async () => {
  // 测试运行器默认跳过样式注入，显式加载实际样式表再检查约束。
  const style = document.createElement("style");
  style.textContent = readFileSync("src/styles.css", "utf8");
  document.head.append(style);
  const history = Array.from({ length: 80 }, (_, index) => ({
    role: index % 2 ? "assistant" : "user",
    content: `历史消息${index}`,
  }));
  vi.stubGlobal("fetch", async (url: string) => {
    if (url === "/api/config")
      return Response.json({
        default_provider: "zhipu",
        settings: {},
        providers: [],
      });
    if (url === "/api/sessions")
      return Response.json([
        { id: "long_chat", title: "长会话", message_count: 80 },
      ]);
    if (url === "/api/sessions/long_chat/messages")
      return Response.json(history);
    throw new Error(`未预期请求：${url}`);
  });
  render(<App />);
  await screen.findByText("历史消息79");
  const container = screen.getByLabelText("聊天记录");
  expect(container.querySelectorAll("article")).toHaveLength(80);
  // JSDOM 不计算真实几何尺寸，此处只验证防溢出样式契约，不替代浏览器视觉验收。
  expect(getComputedStyle(container).minHeight).toBe("0");
  expect(getComputedStyle(container).overflow).toBe("auto");
  expect(getComputedStyle(container.closest(".app")!).gridTemplateRows).toBe(
    "minmax(0, 1fr)",
  );
  expect(getComputedStyle(container.closest(".app")!).position).toBe("fixed");
  expect(getComputedStyle(container.closest(".app")!).overflow).toBe("clip");
  expect(getComputedStyle(container.closest(".main")!).gridTemplateRows).toBe(
    "auto minmax(0, 1fr) auto",
  );
  const composer = screen
    .getByRole("textbox", { name: "输入问题" })
    .closest(".composer-area")!;
  expect(getComputedStyle(composer).flexShrink).toBe("0");
  expect(
    getComputedStyle(container.querySelector(".user .message-body")!).maxWidth,
  ).toBe("85%");
  style.remove();
});

it("点击旧历史时清理上一会话，长 Markdown 正常显示且不新建或改写历史", async () => {
  let finishHistory!: (value: Response) => void;
  const pendingHistory = new Promise<Response>((resolve) => {
    finishHistory = resolve;
  });
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method) throw new Error("历史浏览不能写入数据");
    if (url === "/api/config")
      return Response.json({
        default_provider: "zhipu",
        settings: {},
        providers: [],
      });
    if (url === "/api/sessions")
      return Response.json([
        { id: "recent", title: "最近会话", message_count: 1 },
        { id: "legacy", title: "旧历史记录", message_count: 2 },
      ]);
    if (url === "/api/sessions/recent/messages")
      return Response.json([{ role: "assistant", content: "上一会话回答" }]);
    if (url === "/api/sessions/legacy/messages") return pendingHistory;
    throw new Error(`未预期请求：${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<App />);
  await screen.findByText("上一会话回答");
  fireEvent.click(screen.getByRole("button", { name: /旧历史记录/ }));
  expect(screen.queryByText("上一会话回答")).toBeNull();
  expect(screen.getByText("正在加载工作区…")).toBeTruthy();
  finishHistory(
    Response.json([
      { role: "user", content: "历史中的问题" },
      {
        role: "assistant",
        content: "## 旧回答标题\n\n" + "历史回答正文。\n\n".repeat(100),
      },
    ]),
  );
  await screen.findByRole("heading", { name: "旧回答标题" });
  expect(screen.getByRole("article", { name: "你的消息" }).textContent).toBe(
    "历史中的问题",
  );
  expect(screen.getByRole("textbox", { name: "输入问题" })).toBeTruthy();
  expect(
    screen.getByRole("button", { name: /新建对话/ }).hasAttribute("disabled"),
  ).toBe(false);
  expect(
    screen
      .getByRole("button", { name: /旧历史记录/ })
      .classList.contains("selected"),
  ).toBe(true);
});
