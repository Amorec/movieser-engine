# @movieser/scheduler

> Idle-time priority queue with pre-emptive deadline scheduling for complex browser UIs.
>
> Zero dependencies · Framework-agnostic · < 2 KB minified

---

> ⚠️ **Research-phase package (WBSO Onderzoekslus II / NLnet M1)**
> API is UNSTABLE until v1.0.0. Do not use in production.

## What (will it) solve

Huidige gangbare libraries (React Scheduler, p-queue, queue-lit) missen *either*:
- Standalone usage (React Scheduler is internal)
- Explicit idle-slice budget pre-emption
- Absolute user-input priority over render/idle work

`@movieser/scheduler` implements a 3-tier priority queue:

| Tier | Priority | Description | Deadline |
|------|----------|-------------|----------|
| P0   | 0        | User-blocking (pointer / keyboard input) | Synchronous < 0.3 ms |
| P1   | 1        | Render-diff patches | `requestIdleCallback` remaining time slice, pre-empt if < 1 ms |
| P2   | 2        | Idle-only analytics / logging | Only if budget >= 2 ms, skipped otherwise |

## Research targets (WBSO H1)

- 99.7% of events dispatched within 1 ms
- 0% missed input events under React Fiber contention
- Benchmark: 10 000 synthetic events vs React Scheduler vs p-queue

## Quick start (placeholder v0.x)

```ts
import { PriorityQueue, Priority } from "@movieser/scheduler";

const scheduler = new PriorityQueue({ debug: true });

// P0 — runs immediately
scheduler.enqueue(Priority.P0_USER_BLOCKING, () => handleKeyDown(e));

// P1 — runs only when idle-slice has time
scheduler.enqueue(Priority.P1_RENDER_DIFF, () => applyUiDiff(diff));

// P2 — best-effort, may be dropped under load
scheduler.enqueue(Priority.P2_IDLE_ONLY, () => logAnalytics(e));
```

## License

Dual-licensed under **MIT OR Apache-2.0**.
Pick whichever you prefer.
