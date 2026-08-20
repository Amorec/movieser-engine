// ============================================================
// @movieser/silent-persistence — Placeholder (WBSO Onderzoekslus III / NLnet M3)
// ============================================================
//
// Stub only. Real implementation is research-track:
//  - Layer 1: Web Worker offload via SharedArrayBuffer + Atomics
//             structuredClone happens OFF the main thread
//  - Layer 2: IndexedDB Write-Ahead Log with delta-batching
//             (min 5 mutaties per transactie)
//  - Layer 3: Background Sync API + last-write-wins conflict resolution
//             (per node, geen modal dialogs, geen notificaties)
//
// Targets:
//  - P95 write latency ≤50 ms (payload ≤10 kB)
//  - 0% rAF-blocking (no frames dropped during write)
//  - iOS Safari 17.x fallback: OPFS (Origin Private File System)
//
// DO NOT write code outside the WBSO research iteration cycle.
// First: document H0/H1 + measurement setup in the WBSO logboek.

export type StorageBackend = "indexeddb" | "opfs" | "memory";

export type ConflictResolution = "last-write-wins" | "first-write-wins" | "manual";

export interface SilentStoreOptions<T> {
  /** Name of the database / OPFS root directory */
  name: string;
  /** Prefer IndexedDB or OPFS. Default: IndexedDB with OPFS fallback for WebKit mobile */
  backend?: StorageBackend | "auto";
  /** Minimum number of mutations before committing a batch WAL transaction. Default 5 */
  batchThreshold?: number;
  /** Background Sync tag (Chrome/Firefox). Default "movieser-wal-sync") */
  syncTag?: string;
  /** Conflict resolution strategy. Default last-write-wins */
  conflict?: ConflictResolution;
  /** Enable worker offload (SharedArrayBuffer). Requires COOP/COEP headers. Default true */
  workerOffload?: boolean;
  /** Debug logging (adds ~1% overhead) */
  debug?: boolean;
}

export interface PersistenceStats {
  backend: StorageBackend;
  writesTotal: number;
  writesP50Ms: number;
  writesP95Ms: number;
  rAFDropsDuringWrite: number;
  pendingInWAL: number;
  driftMsSinceServerSync: number;
}

export interface WriteResult {
  ok: true;
  latencyMs: number;
  committed: boolean;
}

export class SilentStore<T extends object> {
  constructor(_options: SilentStoreOptions<T>) {
    // TODO M3 (WBSO Onderzoekslus III, Iteratie 1)
    // Implementatie alleen NA H0/H1 documentatie + experiment setup
  }

  async put(id: string, value: Partial<T>): Promise<WriteResult> {
    void id;
    void value;
    // Placeholder synchronous ack (geen echte write!)
    return Promise.resolve({ ok: true, latencyMs: 0, committed: false });
  }

  async get(id: string): Promise<T | undefined> {
    void id;
    return Promise.resolve(undefined);
  }

  /** Returns true if there are un-synced mutations in WAL */
  hasPending(): boolean {
    return false;
  }

  get stats(): PersistenceStats {
    return {
      backend: "memory",
      writesTotal: 0,
      writesP50Ms: 0,
      writesP95Ms: 0,
      rAFDropsDuringWrite: 0,
      pendingInWAL: 0,
      driftMsSinceServerSync: 0,
    };
  }

  /** Trigger sync with Background Sync API — no-op if offline. No UI feedback */
  triggerBackgroundSync(): Promise<void> {
    return Promise.resolve();
  }

  close(): void {
    // teardown worker, close DB
  }
}
