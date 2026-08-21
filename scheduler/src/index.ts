// ============================================================
// @movieser/scheduler — Priority Scheduler (M1 WBSO / NLnet)
// ============================================================
// Real implementation replacing the stub. Keeps full backwards compat
// with the 3-tier 0/1/2 Priority enum (P0_USER_BLOCKING etc.) from the
// previous placeholder. Adds 4th tier P3_IDLE_GC, 28 typed JobType union,
// binary min-heap, per-job retry with exponential backoff, lazy cancellation,
// and adapter registry for dual-fallback providers.
//
// Zero runtime dependencies. Framework agnostic (Node 18+ / modern browser).
// ============================================================

export { BinaryHeap, compareHeapKey } from './heap.js';
export * from './types.js';

import type {
  AdapterResult,
  EnqueuedJob,
  HeapKey,
  HeapNode,
  JobAdapter,
  JobPayload,
  JobStatus,
  PriorityLevel,
  RetryPolicy,
  SchedulerEvents,
  SchedulerHandle,
  SchedulerOptions,
  SchedulerStats,
} from './types.js';
import {
  DEFAULT_RETRY,
  PRIORITY_NAME,
  Priority,
} from './types.js';
import { BinaryHeap, compareHeapKey } from './heap.js';

// Minimal uuid v4 (crypto.randomUUID available in Node 19+ and all modern browsers).
// Falls back to a high-entropy monotonic id if unavailable (SSR/tests).
function makeJobId(): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = globalThis as any;
    if (g.crypto?.randomUUID) return g.crypto.randomUUID();
  } catch {
    /* ignore */
  }
  const hrt = (typeof process !== 'undefined' && process.hrtime?.bigint?.()) || BigInt(Date.now());
  return `j_${hrt.toString(36)}_${Math.floor(Math.random() * 1e9).toString(36)}`;
}

// Back compat QueueOptions from the stub (debug flag, idleFallbackDelayMs)
export interface QueueOptions extends SchedulerOptions {
  debug?: boolean;
  idleFallbackDelayMs?: number;
}

export class PriorityQueue implements SchedulerHandle {
  private readonly _opts: {
    defaultConcurrency: number;
    tickMs: number;
    idleGcIntervalMs: number;
    maxConcurrencyPerPriority: Record<PriorityLevel, number>;
    retryDefaults: RetryPolicy;
    tokenBucket: { capacityPerUser: number; refillRatePerSecPerUser: number } | undefined;
    debug: boolean;
  };

  private readonly _heap = new BinaryHeap<unknown, unknown>();
  private readonly _byId = new Map<string, EnqueuedJob<unknown, unknown>>();
  private readonly _byGroup = new Map<string, Set<string>>();
  private readonly _running = new Set<string>();
  private readonly _adapters = new Map<string, JobAdapter<unknown, unknown>>();
  private readonly _fallbackAdapters: Array<JobAdapter<unknown, unknown>> = [];
  private readonly _listeners: {
    [K in keyof SchedulerEvents]?: Set<(p: SchedulerEvents[K]) => void | Promise<void>>;
  } = {};

  // Stats counters
  private _sEnqueued = 0;
  private _sProcessed = 0;
  private _sFailed = 0;
  private _sCancelled = 0;
  private _sRetried = 0;
  private _sPreEmpted = 0;
  private _sMissed = 0;
  private _consecutiveIdleTicks = 0;

  private _timer: ReturnType<typeof setInterval> | null = null;
  private _stopped = false;
  private _stopDrainPromise: Promise<void> | null = null;
  private _idleGcTick = 0;
  private readonly _tbRefillAt = new Map<string, { tokens: number; updatedAtMs: number }>();

  constructor(options: QueueOptions = {}) {
    const perPrio = (options.maxConcurrencyPerPriority ?? {}) as Record<
      PriorityLevel,
      number
    >;
    this._opts = {
      defaultConcurrency: options.defaultConcurrency ?? 8,
      tickMs: options.tickMs ?? 4,
      idleGcIntervalMs: options.idleGcIntervalMs ?? 60_000,
      maxConcurrencyPerPriority: {
        0: perPrio[0] ?? 4,
        1: perPrio[1] ?? (options.defaultConcurrency ?? 8),
        2: perPrio[2] ?? 2,
        3: perPrio[3] ?? 1,
      },
      retryDefaults: { ...DEFAULT_RETRY, ...(options.retryDefaults ?? {}) },
      tokenBucket: options.tokenBucket,
      debug: options.debug ?? false,
    };
    this._ensureTimer();
  }

  // ------------------------------------------------------------------
  // Public API (SchedulerHandle)
  // ------------------------------------------------------------------

  /**
   * Enqueue a job with the modern typed JobPayload contract.
   * Returns the EnqueuedJob immediately (does not wait for completion).
   */
  async enqueue<TInput, TResult = unknown>(
    payload: JobPayload<TInput>,
  ): Promise<EnqueuedJob<TInput, TResult>>;

  /**
   * LEGACY BACK-COMPAT OVERLOAD (v0 placeholder API) — signature:
   *   q.enqueue(priority: PriorityLevel, fn: () => TResult | Promise<TResult>)
   *
   * Waits for job completion and returns TResult directly (behaves like
   * the placeholder stub).
   *
   * @deprecated Use the modern `enqueue(JobPayload)` signature instead.
   */
  async enqueue<TResult>(
    priority: PriorityLevel,
    fn: () => TResult | Promise<TResult>,
  ): Promise<TResult>;

  async enqueue<TInput, TResult>(
    first: JobPayload<TInput> | PriorityLevel,
    second?: () => TResult | Promise<TResult>,
  ): Promise<EnqueuedJob<TInput, TResult> | TResult> {
    // ---------- LEGACY BRANCH: enqueue(priority, fn) ----------
    if (typeof first === 'number' && typeof second === 'function') {
      const priority = first as PriorityLevel;
      const fn = second;
      const legacyAdapterName = `__legacy_fn_${makeJobId()}`;
      let resolveResult!: (value: TResult) => void;
      let rejectResult!: (err: unknown) => void;
      const resultPromise = new Promise<TResult>((res, rej) => {
        resolveResult = res;
        rejectResult = rej;
      });

      const legacyAdapter: JobAdapter<unknown, unknown> = {
        type: 'fallback',
        name: legacyAdapterName,
        supports: (job: JobPayload) => job.type === 'generic-http-task' && (job.input as Record<string, unknown>).__legacy === legacyAdapterName,
        execute: async () => {
          try {
            const r = await fn();
            return { success: true, result: r, retryable: false };
          } catch (e) {
            return {
              success: false,
              retryable: false,
              error: { code: 'LEGACY_FN_THREW', message: e instanceof Error ? e.message : String(e) },
            };
          }
        },
      };
      this.registerAdapter(legacyAdapter);

      const unsubComplete = this.on('job:complete', (j) => {
        if ((j.input as Record<string, unknown>).__legacy === legacyAdapterName) {
          unsubComplete();
          unsubFail();
          this.unregisterAdapter(legacyAdapterName);
          resolveResult(j.result as TResult);
        }
      });
      const unsubFail = this.on('job:fail', (j) => {
        if ((j.input as Record<string, unknown>).__legacy === legacyAdapterName) {
          unsubComplete();
          unsubFail();
          this.unregisterAdapter(legacyAdapterName);
          rejectResult(new Error(j.error?.message ?? 'legacy job failed'));
        }
      });

      const payload: JobPayload<{ __legacy: string; fn: string }> = {
        userId: 'legacy-user',
        type: 'generic-http-task',
        priority,
        input: { __legacy: legacyAdapterName, fn: fn.toString().slice(0, 32) },
        retry: { maxAttempts: 1, backoffMs: 0, backoffMultiplier: 1, retryOnStatusCodes: [] },
        timeoutMs: 5 * 60 * 1000,
      };
      const job = this._enqueueInner<TInput, TResult>(payload as unknown as JobPayload<TInput>);
      this._emit('job:queued', job as EnqueuedJob);
      void this._tick();
      return resultPromise;
    }

    // ---------- MODERN BRANCH: enqueue(JobPayload) ----------
    const payload = first as JobPayload<TInput>;
    const job = this._enqueueInner<TInput, TResult>(payload);
    this._emit('job:queued', job as EnqueuedJob);
    void this._tick();
    return job;
  }

  async cancel(jobId: string): Promise<boolean> {
    const j = this._byId.get(jobId);
    if (!j) return false;
    if (j.status === 'completed' || j.status === 'cancelled' || j.status === 'failed') return false;
    j.status = 'cancelled';
    j.finishedAtMs = Date.now();
    this._sCancelled += 1;
    if (this._running.has(jobId)) this._sPreEmpted += 1;
    this._running.delete(jobId);
    this._emit('job:cancel', j);
    return true;
  }

  async cancelGroup(jobGroup: string): Promise<number> {
    const set = this._byGroup.get(jobGroup);
    if (!set) return 0;
    let n = 0;
    for (const id of Array.from(set)) {
      if (await this.cancel(id)) n += 1;
    }
    return n;
  }

  async getStatus(jobId: string): Promise<EnqueuedJob | undefined> {
    return this._byId.get(jobId);
  }

  get snapshot(): SchedulerStats {
    const countsByPriority: Record<PriorityLevel, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
    const countsByStatus: Record<JobStatus, number> = {
      queued: 0,
      running: 0,
      paused: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      'retry-scheduled': 0,
    };
    let oldest: number | null = null;
    const now = Date.now();
    this._heap.forEach((n) => {
      const j = n.job;
      if (j.status === 'queued' || j.status === 'retry-scheduled') {
        countsByPriority[j.priority] += 1;
        countsByStatus[j.status] += 1;
        const age = now - j.createdAtMs;
        if (oldest === null || age > oldest) oldest = age;
      }
    });
    for (const id of this._running) {
      const j = this._byId.get(id);
      if (j) {
        countsByStatus[j.status] += 1;
        countsByPriority[j.priority] += 1;
      }
    }
    // include completed/failed/cancelled from _byId counts (cheap)
    for (const j of this._byId.values()) {
      if (
        j.status === 'completed' ||
        j.status === 'failed' ||
        j.status === 'cancelled' ||
        j.status === 'paused'
      ) {
        countsByStatus[j.status] += 1;
      }
    }

    return {
      enqueued: this._sEnqueued,
      processed: this._sProcessed,
      failed: this._sFailed,
      cancelled: this._sCancelled,
      retried: this._sRetried,
      preEmpted: this._sPreEmpted,
      missedDeadlines: this._sMissed,
      countsByPriority,
      countsByStatus,
      oldestWaitingMs: oldest,
    };
  }

  get stats(): SchedulerStats {
    return this.snapshot;
  }

  on<K extends keyof SchedulerEvents>(
    event: K,
    cb: (payload: SchedulerEvents[K]) => void | Promise<void>,
  ): () => void {
    const set = (this._listeners[event] as
      | Set<(p: SchedulerEvents[K]) => void | Promise<void>>
      | undefined) ?? new Set();
    set.add(cb);
    (this._listeners[event] as unknown as typeof set) = set;
    return () => {
      set.delete(cb);
    };
  }

  registerAdapter(adapter: JobAdapter<unknown, unknown>): void {
    this._adapters.set(adapter.name, adapter);
    if (adapter.type === 'fallback') this._fallbackAdapters.push(adapter);
  }

  unregisterAdapter(name: string): boolean {
    const a = this._adapters.get(name);
    if (!a) return false;
    this._adapters.delete(name);
    const fbi = this._fallbackAdapters.indexOf(a);
    if (fbi >= 0) this._fallbackAdapters.splice(fbi, 1);
    return true;
  }

  clear(priority?: PriorityLevel): void {
    if (priority === undefined) {
      this._heap.clear();
      this._byId.clear();
      this._byGroup.clear();
      this._running.clear();
      return;
    }
    this._heap.rebuildFiltered((n) => n.job.priority !== priority);
    for (const [id, j] of this._byId) {
      if (j.priority === priority && !this._running.has(id)) {
        this._byId.delete(id);
      }
    }
  }

  dispose(): void {
    void this.stop(false);
  }

  async stop(drain: boolean): Promise<void> {
    this._stopped = true;
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    if (drain) {
      if (!this._stopDrainPromise) {
        this._stopDrainPromise = this._drainAll();
      }
      return this._stopDrainPromise;
    }
    return;
  }

  // ------------------------------------------------------------------
  // Internals
  // ------------------------------------------------------------------

  private _enqueueInner<TInput, TResult>(
    payload: JobPayload<TInput>,
  ): EnqueuedJob<TInput, TResult> {
    const retry: RetryPolicy = { ...this._opts.retryDefaults, ...(payload.retry ?? {}) };
    const createdAtMs = Date.now();
    const job = {
      jobId: makeJobId(),
      userId: payload.userId,
      ...(payload.productionId !== undefined && { productionId: payload.productionId }),
      ...(payload.sceneId !== undefined && { sceneId: payload.sceneId }),
      type: payload.type,
      priority: payload.priority,
      input: payload.input,
      retry,
      timeoutMs: payload.timeoutMs ?? 15 * 60 * 1000,
      ...(payload.jobGroup !== undefined && { jobGroup: payload.jobGroup }),
      createdAtMs,
      attempt: 1,
      status: 'queued' as const,
    } as unknown as EnqueuedJob<TInput, TResult>;
    this._sEnqueued += 1;
    this._byId.set(job.jobId, job as EnqueuedJob);
    if (job.jobGroup) {
      const s = this._byGroup.get(job.jobGroup) ?? new Set();
      s.add(job.jobId);
      this._byGroup.set(job.jobGroup, s);
    }
    const key: HeapKey = [job.priority, job.createdAtMs] as const;
    this._heap.insert({ key, jobId: job.jobId, job: job as EnqueuedJob });
    return job;
  }

  private _ensureTimer(): void {
    if (this._timer || this._stopped) return;
    this._timer = setInterval(() => {
      void this._tick();
    }, this._opts.tickMs);
    if (typeof this._timer === 'object' && 'unref' in this._timer) {
      // Allow Node process to exit cleanly if scheduler is idle w/o activity
      (this._timer as NodeJS.Timeout).unref?.();
    }
  }

  private _emit<K extends keyof SchedulerEvents>(
    event: K,
    payload: SchedulerEvents[K],
  ): void {
    const set = this._listeners[event] as unknown as
      | Set<(p: SchedulerEvents[K]) => void | Promise<void>>
      | undefined;
    if (!set || set.size === 0) return;
    for (const cb of set) {
      try {
        const r = cb(payload);
        if (r && typeof (r as Promise<unknown>).catch === 'function') {
          (r as Promise<unknown>).catch(() => {
            /* listener errors are swallowed to avoid breaking dispatcher */
          });
        }
      } catch {
        /* ignore */
      }
    }
  }

  private _runningPerPriority(p: PriorityLevel): number {
    let n = 0;
    for (const id of this._running) {
      const j = this._byId.get(id);
      if (j && j.priority === p) n += 1;
    }
    return n;
  }

  private _consumeToken(userId: string): boolean {
    const tb = this._opts.tokenBucket;
    if (!tb) return true;
    const now = Date.now();
    const cur = this._tbRefillAt.get(userId) ?? {
      tokens: tb.capacityPerUser,
      updatedAtMs: now,
    };
    const elapsedSec = (now - cur.updatedAtMs) / 1000;
    const addTokens = elapsedSec * tb.refillRatePerSecPerUser;
    let tokens = Math.min(tb.capacityPerUser, cur.tokens + addTokens);
    if (tokens < 1) {
      cur.tokens = tokens;
      cur.updatedAtMs = now;
      this._tbRefillAt.set(userId, cur);
      return false;
    }
    tokens -= 1;
    this._tbRefillAt.set(userId, { tokens, updatedAtMs: now });
    return true;
  }

  private async _tick(): Promise<void> {
    if (this._stopped) return;
    const tickStart = Date.now();
    let dispatched = 0;
    // Scan up to 2 heap batches per tick to avoid long blocking ticks
    for (let pass = 0; pass < 2; pass++) {
      let next = this._heap.peek();
      while (next) {
        const j = next.job;
        // lazy-cancel + lazy-retry-delay check
        if (j.status === 'cancelled' || j.status === 'failed' || j.status === 'completed') {
          this._heap.extractMin();
          next = this._heap.peek();
          continue;
        }
        if (j.status === 'retry-scheduled') {
          const scheduled = j.scheduledAtMs ?? 0;
          if (scheduled > tickStart) break; // no older jobs to process — heap is FIFO sorted
          j.status = 'queued';
        }
        if (j.status !== 'queued') {
          this._heap.extractMin();
          next = this._heap.peek();
          continue;
        }
        // concurrency gate per priority
        if (this._runningPerPriority(j.priority) >= this._opts.maxConcurrencyPerPriority[j.priority]) {
          // Can't run at this priority this tick; try lower priority? Heap order means next
          // will be >= priority. So no point peeking further.
          break;
        }
        // Token bucket (user fair share)
        if (!this._consumeToken(j.userId)) {
          break;
        }
        this._heap.extractMin();
        void this._executeJob(next);
        dispatched += 1;
        next = this._heap.peek();
      }
    }

    // idle GC sweep (every ~60s while ticking)
    this._idleGcTick += this._opts.tickMs;
    if (this._idleGcTick >= this._opts.idleGcIntervalMs) {
      this._idleGcTick = 0;
      this._heap.rebuildFiltered((n) => n.job.status !== 'cancelled' && n.job.status !== 'completed' && n.job.status !== 'failed');
    }

    const budgetExceeded = Date.now() - tickStart - this._opts.tickMs;
    if (budgetExceeded > 10) {
      this._emit('scheduler:tick-overrun', { tickMs: this._opts.tickMs, budgetExceededMs: budgetExceeded });
    }
    if (dispatched === 0 && this._running.size === 0) {
      this._consecutiveIdleTicks += 1;
      if (this._consecutiveIdleTicks % 256 === 0) {
        this._emit('scheduler:idle', { consecutiveIdleTicks: this._consecutiveIdleTicks });
      }
    } else {
      this._consecutiveIdleTicks = 0;
    }
  }

  private async _executeJob(heapNode: HeapNode): Promise<void> {
    const j = heapNode.job;
    if (j.status !== 'queued') return;
    j.status = 'running';
    j.startedAtMs = Date.now();
    this._running.add(j.jobId);
    this._emit('job:start', j);

    const ctrl = new AbortController();
    const timeoutTimer = setTimeout(() => ctrl.abort('timeout'), j.timeoutMs);

    let result: AdapterResult<unknown> | null = null;
    let usedAdapter: JobAdapter | null = null;
    try {
      const adapters = this._pickAdapters(j);
      for (let i = 0; i < adapters.length; i++) {
        const ad = adapters[i]!;
        try {
          result = await ad.execute(j, ctrl.signal);
          usedAdapter = ad;
          if (result.success) {
            // primary/fallback success with different adapters
            if (i > 0) {
              this._emit('adapter:fallback', {
                jobId: j.jobId,
                primary: adapters[0]!.name,
                fallback: ad.name,
              });
            }
            break;
          }
          // success=false → continue to NEXT adapter (fallback) regardless of retryable.
          // retryable flag only determines whether the ENTIRE job is re-enqueued for retry
          // after ALL adapters in the chain have been exhausted without success.
        } catch (err) {
          result = {
            success: false,
            retryable: true,
            error: {
              code: 'ADAPTER_THREW',
              message: err instanceof Error ? err.message : String(err),
            },
          };
          usedAdapter = ad;
          // exception: also continue to next adapter if any
        }
      }
      if (!result) {
        result = {
          success: true,
          result: undefined,
          retryable: false,
        };
      }
    } finally {
      clearTimeout(timeoutTimer);
    }

    this._running.delete(j.jobId);
    j.finishedAtMs = Date.now();

    if (result.success) {
      j.status = 'completed';
      j.result = result.result;
      this._sProcessed += 1;
      this._emit('job:complete', j);
      return;
    }

    const err = result.error ?? { code: 'UNKNOWN', message: 'adapter failed silently' };
    // retry check
    const retryStatus = !result.retryable
      ? false
      : (result.error?.code !== undefined
          ? !j.retry.retryOnStatusCodes ||
            j.retry.retryOnStatusCodes.includes(parseInt(result.error.code, 10) || 0)
          : true);
    if (j.attempt < j.retry.maxAttempts && retryStatus) {
      j.attempt += 1;
      const delay = Math.round(
        j.retry.backoffMs * Math.pow(j.retry.backoffMultiplier, j.attempt - 1),
      );
      j.status = 'retry-scheduled';
      j.scheduledAtMs = Date.now() + delay;
      this._sRetried += 1;
      // re-insert: key priority is kept, createdAtMs kept = maintains place but delayed via scheduledAtMs gate in tick
      this._heap.insert({
        key: [j.priority, j.createdAtMs] as HeapKey,
        jobId: j.jobId,
        job: j,
      });
      this._emit('job:retry', { ...j, nextAttemptAtMs: j.scheduledAtMs } as SchedulerEvents['job:retry']);
      return;
    }

    j.status = 'failed';
    j.error = err;
    this._sFailed += 1;
    this._emit('job:fail', { ...j, error: j.error } as SchedulerEvents['job:fail']);
    if (j.timeoutMs && j.finishedAtMs - j.startedAtMs >= j.timeoutMs) {
      this._sMissed += 1;
    }
    void usedAdapter; // referenced for future adapter-scoped metrics
  }

  private _pickAdapters(j: EnqueuedJob): Array<JobAdapter<unknown, unknown>> {
    const result: Array<JobAdapter<unknown, unknown>> = [];
    const byType: Array<JobAdapter<unknown, unknown>> = [];
    for (const ad of this._adapters.values()) {
      if (ad.type === 'fallback') continue;
      const t = ad.type;
      const match = Array.isArray(t) ? t.includes(j.type) : t === j.type;
      if (match && ad.supports(j)) byType.push(ad);
    }
    byType.sort((a, b) => a.name.localeCompare(b.name));
    result.push(...byType);
    for (const fb of this._fallbackAdapters) {
      if (fb.supports(j)) result.push(fb);
    }
    return result;
  }

  private async _drainAll(): Promise<void> {
    const poll = (): Promise<void> =>
      new Promise((resolve) => {
        const iv = setInterval(() => {
          const snap = this.snapshot;
          const busy = this._heap.size > 0 || this._running.size > 0;
          if (!busy) {
            clearInterval(iv);
            resolve();
          }
          void snap;
        }, 25);
      });
    await poll();
  }
}

// Re-export named Priority alias map (exact names used by placeholder tests)
export {
  Priority as PriorityLevel_Enum_DO_NOT_USE,
  /** @deprecated Use named levels via exported type `PriorityLevel` and const `Priority.*` directly. */
  PRIORITY_NAME,
};
// Keep top-of-file `export * from './types.js'` above already does it; legacy compatibility stubs:
export const P0_USER_BLOCKING = Priority.P0_USER_BLOCKING;
export const P1_RENDER_DIFF = Priority.P1_RENDER_DIFF;
export const P2_IDLE_ONLY = Priority.P2_IDLE_ONLY;

// Unused helper silences some strict-Tree-shaker bundlers when compareHeapKey is tree-shaken.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const _ = compareHeapKey;
void _;
