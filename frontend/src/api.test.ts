import { describe, expect, it, vi, afterEach } from "vitest";
import { EventParser, streamChat } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("SSE事件解析", () => {
  it("UTF8字节被切在中文中间也不会乱码", async () => {
    const bytes = new TextEncoder().encode(
      'event: token\ndata: {"text":"中文证据"}\n\nevent: done\ndata: {}\n\n',
    );
    vi.stubGlobal(
      "fetch",
      async () =>
        new Response(
          new ReadableStream({
            start(controller) {
              for (const byte of bytes)
                controller.enqueue(new Uint8Array([byte]));
              controller.close();
            },
          }),
        ),
    );
    const events: { event: string; data: Record<string, unknown> }[] = [];
    await streamChat("chat_test", "问题", "zhipu", (event) =>
      events.push(event),
    );
    expect(events[0].data.text).toBe("中文证据");
  });
  it("跨分包保留中文、注释与多个事件", () => {
    const parser = new EventParser();
    expect(parser.push(": keepalive\n\nevent: tok")).toEqual([]);
    expect(parser.push('en\r\ndata: {"text":"中文"}\r\n')).toEqual([]);
    expect(parser.push("\r\nevent: done\ndata: {}\n\n")).toEqual([
      { event: "token", data: { text: "中文" } },
      { event: "done", data: {} },
    ]);
  });
  it("没有终结事件不能算完成", async () => {
    vi.stubGlobal(
      "fetch",
      async () => new Response('event: token\ndata: {"text":"部分"}\n\n'),
    );
    await expect(
      streamChat("chat_test", "问题", "zhipu", () => {}),
    ).rejects.toThrow("连接提前结束");
  });
  it("保留明确的后端错误事件", async () => {
    vi.stubGlobal(
      "fetch",
      async () =>
        new Response('event: error\ndata: {"message":"模型超时"}\n\n'),
    );
    const events: unknown[] = [];
    await streamChat("chat_test", "问题", "zhipu", (event) =>
      events.push(event),
    );
    expect(events).toEqual([{ event: "error", data: { message: "模型超时" } }]);
  });
});
