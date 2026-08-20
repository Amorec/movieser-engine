# @movieser/ctm-engine

> Context-Triggered Mounting (CTM) engine for media-rich canvas UIs.
>
> Mounts only **max 5% of total nodes** on pointer-down selection + 40px radius
> prediction. Includes a synchronous `DisposeHookRegistry` that deterministically
> releases Blob URLs, AudioContexts, WebWorkers, and WASM instances on deselect
> — eliminating the 120+ MB/h heap-growth leak common in long editing sessions.

---

> ⚠️ **Research-phase package (WBSO Onderzoekslus I / NLnet M2)**.
> Do not use in production until v1.0.0.

## Problem statement (technical)

Standard React lazy / Suspense / virtualisation libraries have three structural
flaws for media-bearing nodes:

1. **Full-mount hydration** — even with `React.memo`, 500+ video nodes cause
   40-120 ms of blocking main-thread time on selection.
2. **No 40px pointer-radius prediction** — selection-mount triggers AFTER
   pointer-down, so users see "flash of unmounted content".
3. **Undeterministic GC of media** — Blob URLs, open AudioContexts and Workers
   attached to unmounted components are not released for 5-30s (or never, under
   Chromium 125 heap pressure). Leads to tab crash after 2 h on 8 GB laptops.

## Architecture

```
                     ┌──────────────────────────────┐
 pointer-down event  │  CTMEngine.activate(nodeId)  │
 ─────────────────▶  │  ├ 40px radius neighbours    │
                     │  ├ 5% cap enforce             │
                     │  └ mount() diff-patch        │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
 pointer-up/deselect │  DisposeHookRegistry          │
 ─────────────────▶  │  ├ Blob.revokeObjectURL()     │
                     │  ├ AudioContext.close()       │
                     │  ├ Worker.terminate()         │
                     │  └ WeakRef-based leak checks  │
                     └──────────────────────────────┘
```

## Research targets (WBSO H1 Onderzoekslus I)

- 60%+ of mount transitions complete in ≤16 ms (P99)
- Heap growth ≤10 MB / hour in a 4-hour 500-node session
- 0 flash-of-unmounted-content in ≥95% of pointer interactions
- Fallback strategy (`display:none` pre-mount in Shadow DOM) validated & compared

## Quick start (v0.x placeholder)

```ts
import { CTMEngine, CTMOptions } from "@movieser/ctm-engine";
import {
  DisposeHookRegistry,
  registerBlobUrl,
} from "@movieser/ctm-engine/dispose-hook";

const engine = new CTMEngine({
  activeNodeCapPercent: 5,
  preFetchRadiusPx: 40,
  unmountStrategy: "eager",
});

// register your nodes with absolute bounding rects
engine.register({ id: "v1", rect: { x: 0, y: 0, width: 640, height: 360 }, mediaBearing: true });

// predict mounts from pointer coords (40px radius)
await engine.onPointerMove(20, 20);
await engine.onSelect("v1");
```

## License

Dual **MIT OR Apache-2.0**. Pick whichever suits you.
