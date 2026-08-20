// ============================================================
// @movieser/ctm-engine — Placeholder (WBSO Onderzoekslus I / NLnet M2)
// ============================================================
//
// Stub only. The real implementation is part of WBSO Onderzoekslus I:
//  - Context-Triggered Mounting (CTM): pointer-down + 40px prediction radius
//  - 5% active-node cap (max N nodes mounted from total canvas set)
//  - Deterministic disposal hook registry (Blob.revoke, AudioContext.close, Worker.terminate)
//  - Target: P99 mount/unmount <=16 ms; heap-growth <=10 MB/h in 4h sessions
//
// Do NOT add features outside the WBSO research iteration cycle.
// Log H0/H1 + measurements in /wbso-logboek/ before code changes.

export interface CTMOptions {
  /** Max % of total nodes actively hydrated (CTM cap, default 5%) */
  activeNodeCapPercent?: number;
  /** Pointer-down prediction radius in px — mounts nodes within this radius. Default 40px */
  preFetchRadiusPx?: number;
  /** Strategy for unmount: eager (<=1 frame), lazy (idle), or display-none (fallback). Default: eager */
  unmountStrategy?: "eager" | "lazy-idle" | "display-none";
  /** Debug — log mount/unmount decisions (2-3% overhead). Off by default */
  debugTrace?: boolean;
}

export type NodeId = string | number;

export interface MountableNode {
  id: NodeId;
  /** Bounding rect (absolute) — used for 40px radius prediction */
  rect: { x: number; y: number; width: number; height: number };
  /** Is media-bearing (video/audio/wasm-worker)? Affects dispose-hook priority */
  mediaBearing?: boolean;
}

export interface CTMStats {
  totalNodes: number;
  mountedCount: number;
  activeCapPercent: number;
  lastMountLatencyMs: number;
  lastUnmountLatencyMs: number;
}

export class CTMEngine {
  constructor(_options: CTMOptions = {}) {
    // TODO M2 (WBSO Onderzoekslus I Iteratie 1)
    // Implementeer pas NA hypothese + experiment-opzet in WBSO logboek
  }

  register(_node: MountableNode): void {
    // Placeholder
  }

  /** Pointer-event trigger — 40px radius pre-fetch */
  onPointerMove(_x: number, _y: number): Promise<void> {
    return Promise.resolve();
  }

  onSelect(_nodeId: NodeId): Promise<void> {
    return Promise.resolve();
  }

  onDeselect(_nodeId: NodeId): Promise<void> {
    return Promise.resolve();
  }

  get stats(): CTMStats {
    return {
      totalNodes: 0,
      mountedCount: 0,
      activeCapPercent: 5,
      lastMountLatencyMs: 0,
      lastUnmountLatencyMs: 0,
    };
  }

  dispose(): void {
    // Dispose all — shutdown sequence
  }
}
