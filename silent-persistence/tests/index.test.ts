import { describe, it, expect, beforeEach } from "vitest";
import { SilentStore } from "../src/index";

describe("@movieser/silent-persistence (placeholder)", () => {
  it("placeholder returns empty reads", async () => {
    const store = new SilentStore<{ title: string }>({ name: "test-placeholder" });
    expect(await store.get("x")).toBeUndefined();
  });

  it("placeholder write returns WriteResult", async () => {
    const store = new SilentStore<{ title: string }>({ name: "test-write" });
    const r = await store.put("x", { title: "hello" });
    expect(r.ok).toBe(true);
  });

  it("placeholder stats indicate memory backend", () => {
    const store = new SilentStore<{ title: string }>({ name: "test-stats" });
    expect(store.stats.backend).toBe("memory");
    expect(store.hasPending()).toBe(false);
  });
});
