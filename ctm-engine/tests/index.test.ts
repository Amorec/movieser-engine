import { describe, it, expect, beforeEach } from "vitest";
import { CTMEngine } from "../src/index";
import { DisposeHookRegistry, registerBlobUrl } from "../src/dispose-hook";

describe("@movieser/ctm-engine (placeholder)", () => {
  it("placeholder CTM stats report zeros", () => {
    const e = new CTMEngine();
    expect(e.stats.totalNodes).toBe(0);
    expect(e.stats.activeCapPercent).toBe(5);
  });

  it("dispose registry accepts and counts blob urls", () => {
    const r = new DisposeHookRegistry();
    // use a fake blob:// string (we will NOT revoke — just test bookkeeping)
    registerBlobUrl(r, "blob:fake-test-1234");
    expect(r.aliveCount).toBe(1);
    expect(r.unregister("blob:fake-test-1234")).toBe(true);
    expect(r.aliveCount).toBe(0);
  });
});
