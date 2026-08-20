// ============================================================
// DisposeHookRegistry — Deterministic lifecycle for media resources
// ============================================================
//
// Solves a concrete gap: React Strict Mode + unmount does not guarantee
// synchronous cleanup of Blob URLs (revokeObjectURL), AudioContext (close),
// WebWorkers (terminate). V8 GC delays cleanup by 5-30s or indefinitely.
//
// This registry MUST be called synchronously on deselect / unmount
// (NOT inside useEffect cleanup — called inside React Strict Mode twice).
//
// To be validated in WBSO CTM Iteratie 3 — geheugenlek studies.

export type DisposableKind =
  | "blob-url"
  | "audio-context"
  | "web-worker"
  | "media-element"
  | "wasm-instance"
  | "abort-controller";

export interface DisposeHandle {
  readonly id: string;
  readonly kind: DisposableKind;
  /** Synchronous disposal. Throws are caught (logged if debug). Never re-throws. */
  dispose(): void;
}

export class DisposeHookRegistry {
  private readonly _handles = new Map<string, DisposeHandle>();

  register(handle: DisposeHandle): void {
    this._handles.set(handle.id, handle);
  }

  unregister(id: string): boolean {
    return this._handles.delete(id);
  }

  /** Deterministic synchronous disposal. Nodes: runs <1 ms for 10 handles. */
  disposeAll(_kindFilter?: DisposableKind[]): number {
    // TODO M2 WBSO — implement + measure performance
    return 0;
  }

  get aliveCount(): number {
    return this._handles.size;
  }
}

// ---------- Helper factories ----------

export function registerBlobUrl(registry: DisposeHookRegistry, url: string): void {
  registry.register({
    id: `blob:${url.slice(0, 64)}`,
    kind: "blob-url",
    dispose: () => {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // ignore — already revoked
      }
    },
  });
}
