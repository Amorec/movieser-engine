import { describe, it, expect, beforeEach } from "vitest";
import { PriorityQueue, Priority } from "../src/index";

describe("@movieser/scheduler (placeholder)", () => {
  let q: PriorityQueue;
  beforeEach(() => {
    q = new PriorityQueue();
  });

  it("placeholder enqueue resolves immediately", async () => {
    const r = await q.enqueue(Priority.P2_IDLE_ONLY, () => 42);
    expect(r).toBe(42);
  });

  it("placeholder stats return zeros", () => {
    expect(q.stats.enqueued).toBe(0);
    expect(q.stats.processed).toBe(0);
  });
});
