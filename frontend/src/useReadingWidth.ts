import { useEffect, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, PointerEvent } from "react";

export const DEFAULT_READING_WIDTH = 960;

export function readingLimits(available: number) {
  // 最大宽度保持1280；边缘额外留白，让拖拽线与正文分离且不压到滚动条。
  const max = Math.max(0, Math.min(1280, available - 192));
  return { min: Math.min(560, max), max };
}

export function clampReadingWidth(width: number, available: number): number {
  const { min, max } = readingLimits(available);
  return Math.min(max, Math.max(min, width));
}

export function useReadingWidth() {
  const main = useRef<HTMLElement>(null);
  const drag = useRef<{
    x: number;
    width: number;
    direction: number;
    pointer: number;
  } | null>(null);
  const [available, setAvailable] = useState(1024);
  const [preferred, setPreferred] = useState(DEFAULT_READING_WIDTH);
  const [resizing, setResizing] = useState(false);
  const width = clampReadingWidth(preferred, available);
  const limits = readingLimits(available);

  useEffect(() => {
    const element = main.current;
    if (!element) return;
    const measure = () => {
      if (element.clientWidth > 0) setAvailable(element.clientWidth);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  function handleProps(side: "left" | "right") {
    const direction = side === "right" ? 1 : -1;
    const stop = () => {
      drag.current = null;
      setResizing(false);
    };
    return {
      role: "separator" as const,
      tabIndex: 0,
      "aria-label": side === "left" ? "左侧调整对话宽度" : "右侧调整对话宽度",
      "aria-orientation": "vertical" as const,
      "aria-valuemin": limits.min,
      "aria-valuemax": limits.max,
      "aria-valuenow": Math.round(width),
      "aria-valuetext": `${Math.round(width)} 像素`,
      onPointerDown(event: PointerEvent<HTMLDivElement>) {
        if (event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = {
          x: event.clientX,
          width,
          direction,
          pointer: event.pointerId,
        };
        setResizing(true);
      },
      onPointerMove(event: PointerEvent<HTMLDivElement>) {
        const start = drag.current;
        if (!start || event.pointerId !== start.pointer) return;
        // 阅读区保持居中，单边移动距离对应整体宽度的两倍变化。
        setPreferred(
          clampReadingWidth(
            start.width + 2 * start.direction * (event.clientX - start.x),
            available,
          ),
        );
      },
      onPointerUp: stop,
      onPointerCancel: stop,
      onLostPointerCapture: stop,
      onDoubleClick: () => setPreferred(DEFAULT_READING_WIDTH),
      onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
          return;
        event.preventDefault();
        const next =
          event.key === "Home"
            ? limits.min
            : event.key === "End"
              ? limits.max
              : width + direction * (event.key === "ArrowRight" ? 40 : -40);
        setPreferred(clampReadingWidth(next, available));
      },
    };
  }

  return {
    main,
    width,
    resizing,
    handleProps,
    style: { "--reading-width": `${width}px` } as CSSProperties,
  };
}
