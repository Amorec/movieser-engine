# MovieSer Core — Open Rendering & Scheduling Engine

> **Context-Triggered Open Rendering & Scheduling Engine** — three independent,
> framework-agnostic TypeScript modules that eliminate UI blocking in complex
> browser applications (video editors, canvas tools, document editors, and
> neuro-inclusive interfaces).
>
> Built by Solhei Solutions, funded in part by NLnet NGI0 Commons Fund (2026)
> and conducted as WBSO R&D project (RVO The Netherlands).

Repository: **https://github.com/Amorec/movieser-engine**

---

## ⚠️ Status: Research phase

All three modules are **UNSTABLE** until v1.0.0 (M4 NLnet / Week 16 WBSO).
Placeholders are intentional — the real implementation follows a
**hypothesis → experiment → measurement → conclusion** research loop as part of
WBSO. Do not submit features outside this loop.

| Milestone | Package | Status |
| --- | --- | --- |
| **M1** (WBSO Onderzoekslus II) | `@movieser/scheduler` | 🧪 Placeholder — research starting |
| **M2** (WBSO Onderzoekslus I)  | `@movieser/ctm-engine` | 🧪 Placeholder — research starting |
| **M3** (WBSO Onderzoekslus III)| `@movieser/silent-persistence` | 🧪 Placeholder — research starting |
| **M4** (Packaging + Docs)      | v1.0.0 stable + demo + benchmarks | 📋 Planned NLnet Deliverable |

---

## The three modules

### 1. `@movieser/scheduler` — idle-time priority queue

```
npm install @movieser/scheduler
```

3-tier queue:
- **P0** — user-blocking input (pointer / keyboard), processed synchronously in < 0.3 ms
- **P1** — render-diff patches, processed inside `requestIdleCallback` time slices (deadline-aware pre-emptive)
- **P2** — idle-only analytics, dropped if budget < 2 ms

Research target: **99.7 % of events dispatched in ≤ 1 ms with 0 missed P0 events under React Fiber contention.**

### 2. `@movieser/ctm-engine` — Context-Triggered Mounting

```
npm install @movieser/ctm-engine
```

- **Lazy-mounts only max 5 % of total canvas node set** on pointer-down selection trigger
- **40 px pointer-radius prediction** pre-fetches neighbours to avoid flash-of-unmounted-content
- **`DisposeHookRegistry`** synchronously releases `Blob:`, `AudioContext`, `Worker`, `WASM` on deselect (100 % deterministic — no V8 GC delays)

Research targets:
- **P99 mount/unmount ≤ 16 ms for ≤ 500 media nodes**
- **Heap growth ≤ 10 MB / hour in 4 h sessions**

### 3. `@movieser/silent-persistence` — non-blocking zero-notice storage

```
npm install @movieser/silent-persistence
```

- Serialization offloaded to **Web Worker** with `SharedArrayBuffer + Atomics` for zero-copy
- **IndexedDB Write-Ahead Log** batch-commits ≥ 5 mutations per transaction to minimize IDB commit blocking
- **Background Sync API** + per-node last-write-wins conflict resolution, **100 % silent** — no spinners, no "Saved!" notifications
- Fallback: **OPFS (Origin Private File System)** for iOS Safari 17.x WebKit

Research targets: **P95 write ≤ 50 ms, 0 % dropped rAF-frames**

---

## Installing locally

```bash
# This repo uses npm workspaces
npm install

# Run all package tests
npm run test:packages

# Build all three modules
npm run build:packages

# Publish to npm (requires login, v1.0.0+)
# npm run publish:packages
```

---

## Open Source Commitment

| Deliverable | Timeline | Format |
| --- | --- | --- |
| Stable npm releases (× 3 modules) | End M4 | npm, MIT + Apache 2.0 |
| Live browser demo | End M4 | StackBlitz embed, React |
| 10 000+ benchmark dataset | End M4 | GitHub Pages D3 dashboard, CC-BY 4.0 |
| TypeDoc API coverage | End M4 | `docs.movieser.com` |
| Technical whitepaper | 6 months post-M4 | movieser.com/onderzoek, arXiv |
| Security & bug fix maintainership | 12 months post-M4 | Repository |

### Governance

- No CLA. Uses **Developer Certificate of Origin (DCO)** sign-off on commits.
- Accepting contributions: Bug fixes only during research phase (M1–M3). Feature PRs deferred to M4 roadmap.
- Governance: Benevolent dictator (Solhei Solutions) until post-M4 SIG.

---

## Alignment with NLnet / NGI0 Commons

| Criterium | How MovieSer-core delivers |
| --- | --- |
| **Open commons** | Zero dependency libraries, permissive dual license, DCO governance |
| **Privacy & trust** | `@movieser/silent-persistence` deliberately ships NO telemetry; Background Sync only when the user's runtime explicitly queues syncs |
| **Inclusion** | Focus-first UX building blocks designed for neurodivergent workers (ADHD, autism, attention challenges). 15-20 % of NL workforce benefits. |
| **Digital sovereignty** | European alternative to closed, cloud-first editors (ByteDance CapCut Web, Canva). Build locally-first. |

---

## Funding

```
NLnet Foundation NGI0 Commons Fund  —  € 20 000  (M1–M4, 400 h @ € 50/h)
RVO WBSO (NL Tax Faci. S&O)        —  Fiscaal    (500 research hrs, separate fiscal claim)
```

No double-declaration. 400 NLnet hours = development + publication.
Remaining 100 WBSO hours = baselines, expanded fallback studies, and cross-browser validation.

---

## Licensing

Dual-licensed at the choice of the licensee:

- **MIT License** (permissive, no patent clause, maximum adoption)
- **Apache License 2.0** (includes explicit patent grants)

See `LICENSE-MIT` and `LICENSE-APACHE`.
