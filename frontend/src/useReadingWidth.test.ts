import { expect, it } from "vitest";
import { clampReadingWidth, readingLimits } from "./useReadingWidth";

it("桌面阅读宽度限制为560至1280像素", () => {
  expect(readingLimits(2000)).toEqual({ min: 560, max: 1280 });
  expect(clampReadingWidth(10, 2000)).toBe(560);
  expect(clampReadingWidth(9999, 2000)).toBe(1280);
});

it("窄屏保留边缘空间，不强行撑到桌面最小宽度", () => {
  expect(readingLimits(400)).toEqual({ min: 208, max: 208 });
  expect(clampReadingWidth(960, 400)).toBe(208);
  expect(clampReadingWidth(960, 0)).toBe(0);
});

it("拉到最大也保留两侧拖拽线与正文之间的空隙", () => {
  expect(readingLimits(1200)).toEqual({ min: 560, max: 1008 });
  expect(clampReadingWidth(9999, 1200)).toBe(1008);
});
