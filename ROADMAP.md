# MovieSer Core — Roadmap

Source of truth for the NLnet + WBSO combined project plan.
Do not add items to M1–M3 unless both research programmes approve.

---

## Milestone 1 · Scheduler architecture & benchmarks

**Deliver:** `@movieser/scheduler@0.1.0-canary`
- [x] Package workspace scaffold (placeholder)
- [ ] Priority queue 3 tiers: P0/P1/P2
- [ ] requestIdleCallback dispatcher w/ deadline pre-empt (P1 drops below 1 ms budget → yield)
- [ ] React Scheduler API conflict study: 10 000 events / 4 runs benchmark
- [ ] Vitest ≥ 95 % coverage
- [ ] Benchmark report (v1): custom queue vs. React Scheduler vs. p-queue

**Uren:** 120 h. **NLnet payout:** € 6 000.

---

## Milestone 2 · Context-Triggered Mounting (CTM) + Dispose Registry

**Deliver:** `@movieser/ctm-engine@0.1.0-canary`
- [x] Package workspace scaffold (placeholder)
- [ ] Pointer-down 40 px pre-fetch radius predictor + 5 % active cap
- [ ] Fallback research: `display:none` visibility toggle (shadow DOM) vs. strict CTM
- [ ] DisposeHookRegistry deterministic cleanup
- [ ] 4-hour, 10-run memory-leak harness report (≤ 10 MB / h growth)
- [ ] Vitest ≥ 95 % coverage

**Uren:** 120 h. **NLnet payout:** € 6 000.

---

## Milestone 3 · Silent Persistence Layer (Worker + WAL)

**Deliver:** `@movieser/silent-persistence@0.1.0-canary`
- [x] Package workspace scaffold (placeholder)
- [ ] SharedArrayBuffer + Atomics Web Worker serializer offload
- [ ] IndexedDB Write-Ahead Log batch commit (≥ 5 mutaties / transactie)
- [ ] iOS Safari 17.x compatibility study: IndexedDB ↔ OPFS fallback
- [ ] Background Sync API integration + last-write-wins conflict resol.
- [ ] Latency report: P50 / P95 / P99 + 0 % dropped rAF-frame validation
- [ ] Vitest ≥ 95 % coverage (w/ fake-indexeddb for node runs)

**Uren:** 100 h. **NLnet payout:** € 5 000.

---

## Milestone 4 · Open Source Packaging, Docs & Demo

**Deliver:** v1.0.0 stable releases + public demo + benchmark dashboard
- [ ] npm v1.0.0 × 3 packages
- [ ] TypeDoc API ≥ 95 % coverage, 4 setup guides
- [ ] Framework adapters examples: React / Vue / Svelte (StackBlitz)
- [ ] Live browser demo (React): canvas w/ 500 nodes + 3 modules working together
- [ ] Benchmark dashboard (10k+ measurements) → GitHub Pages
- [ ] Release notes + blog posts (Dev.to, Hacker News, DTComm)
- [ ] Whitepaper draft v1

**Uren:** 60 h. **NLnet payout:** € 3 000.

---

## Post-M4 (community roadmap, open for voting)

| Priority | Topic | Champion needed? |
| --- | --- | --- |
| ★★★ | React bindings (hooks) + Next.js SSR-safe wrapper | Yes |
| ★★★ | Vue / Svelte bindings | Yes |
| ★★☆ | WebGPU offload variant (dispose hook / batch serializer) | Yes |
| ★★☆ | OPFS-first backend (default flip for WebKit 18+) | Yes |
| ★☆☆ | CRDT layer (Yjs / Automerge) for multi-user concurrent editing | Yes |
| ★☆☆ | WASM GC tracing helper (Rust / wasm-bindgen) for >2 000 nodes | Yes |

---

## Quarterly SIG · Browser Performance & Inclusive Editing (post-M4)

First meeting: 3 months after M4 stable release.
Agenda: review last quarter benchmarks, select next 2 roadmap topics,
elect community maintainers.
