# @movieser/silent-persistence

> Non-blocking persistence layer for complex browser UIs (video / canvas /
> real-time editors).
>
> Writes serialized data OFF the main thread via Web Worker + SharedArrayBuffer
> / Atomics. Then commits to an **IndexedDB Write-Ahead Log** (WAL) that batches
> mutations to avoid rAF drops. Synchronizes to the server via the Background Sync
> API with per-node **last-write-wins conflict resolution — ZERO notifications.
> Zero user-visible UI feedback (no "Saved!" toast or spinner). 100% silent persistence.

---

> ⚠️ **Research-phase package (WBSO Onderzoekslus III / NLnet M3)**.
> Do not use in production until v1.0.0.

## Why

The core problem

Standard IndexedDB wrappers (Dexie.js, PouchDB, idb-keyval) block the main thread
during the `transaction.commit()` phase. In **iOS Safari 17.x WebKit** this
produces **≥5%** of **>50 ms latency for payloads ≤ 10 kB — causing dropped
audio glitches in concurrentAudioContext`s and stuttering video playback.

## Architecture

```
Main thread (rAF-loop, 0% blocking goal)
   │
   └── put(id, value)
        │ SharedArrayBuffer + Atomics.notify() ▸ │
        │                                      ▼
        │ Worker thread (serializer + batcher)
        │   1. structuredClone (1 kB -> 10 kB
        │   2. Appends to in-memory WAL buffer
        │   3. Batches ≥5 mutaties → min
        │
        └── Atomics.waitAsync commit
                    │
                    ▼
            IndexedDB / OPFS
   (commit alleen als er 5+ mutaties in batch)
                    │
                    ▼
      Background Sync API (indien offline)
      └── last-write-wins per node
```

## Research targets (WBSO H1)

| Metriek | Doel |
| --- | --- |
| Write P50 | < 10 ms |
| Write P95 | < 50 ms |
| rAF drops under writes | 0 % |
| Sync drift (after reconnect) | < 200 ms |
| iOS Safari 17.x write latency P95 | < 80 ms |

## Quick start (v0.x placeholder)

```ts
import { SilentStore } from "@movieser/silent-persistence";

type Scene = { title: string; nodes: any[]; updatedAt: number };

const store = new SilentStore<Scene>({
  name: "movieser-production-v1",
  backend: "auto",           // IndexedDB met OPFS fallback op iOS WebKit
  batchThreshold: 5,         // batch ≥5 mutaties / transactie
  conflict: "last-write-wins",
  workerOffload: true,
});

await store.put("scene-1", { title: "Intro", nodes: [], updatedAt: Date.now() });
const scene = await store.get("scene-1");
```

## License

Dual **MIT OR Apache-2.0**.
