// ============================================================
// @movieser/scheduler — Data Contract (Types)
// ============================================================
// WBSO Milestone 1 | NLnet M1 (120u) | RVO S&O
// Dual License: MIT OR Apache-2.0
// ============================================================

// ------------------------------------------------------------
// 1. Priority Levels (4-tier, tuple-sort compatible)
// ------------------------------------------------------------
// Backwards compat: existing `Priority.P0_USER_BLOCKING: 0` etc.
// maps 1:1 to the 4-tier system below (P0=urgent, P1=high, P2=bg)
export type PriorityLevel = 0 | 1 | 2 | 3;

export const Priority = {
  /** User-blocking interactive: user sees spinner / waits */
  URGENT: 0 as const,
  P0_USER_BLOCKING: 0 as const,

  /** AI-image, video-chunk render, voice, whisper-asr, flux */
  HIGH: 1 as const,
  P1_RENDER_DIFF: 1 as const,

  /** persistence-sync, tag-assets, music, compaction */
  BACKGROUND: 2 as const,
  P2_IDLE_ONLY: 2 as const,

  /** GC, cold backups, offline sync queue drains */
  IDLE: 3 as const,
  P3_IDLE_GC: 3 as const,
} as const;

/** Reverse lookup for logs/metrics (cheap enum-free) */
export const PRIORITY_NAME: Record<PriorityLevel, string> = {
  0: 'urgent',
  1: 'high',
  2: 'background',
  3: 'idle',
};

// ------------------------------------------------------------
// 2. Job Type Union (28 types, incl. toekomstige MCP/Luma/Seedance)
// ------------------------------------------------------------
export type JobType =
  // Render pipeline (operationeel vandaag in ser-production-ct)
  | 'first-cut-render'
  | 'remotion-render-chunk'
  // AI beeld
  | 'ai-image-flux-hf'
  | 'ai-image-flux-replicate'
  | 'ai-image-sdxl'
  // AI video (huidig + toekomst)
  | 'ai-video-svd-replicate'
  | 'luma-video'
  | 'luma-scenes-render'
  | 'seedance-video'
  // Audio
  | 'voice-elevenlabs'
  | 'music-mureka'
  | 'whisper-asr'
  | 'tag-ai-assets'
  // Data persistence (M3 — silent-persistence)
  | 'persistence-sync'
  | 'idb-compaction'
  // MCP bridge / GPU on-demand (M2.5 — Septiembre GPU)
  | 'mcp-bridge-generic'
  | 'comfyui-image'
  | 'comfyui-video'
  | 'vastai-boot-gpu-pod'
  | 'vastai-shutdown-gpu-pod'
  | 'ffmpeg-concat-chunks'
  // Overig / catch-all
  | 'generic-http-task';

// ------------------------------------------------------------
// 3. Job Lifecycle & Retry
// ------------------------------------------------------------
export type JobStatus =
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'retry-scheduled';

export interface RetryPolicy {
  maxAttempts: number;
  backoffMs: number;
  backoffMultiplier: number;
  retryOnStatusCodes?: readonly number[];
}

export const DEFAULT_RETRY: RetryPolicy = {
  maxAttempts: 3,
  backoffMs: 1_000,
  backoffMultiplier: 2.0,
  retryOnStatusCodes: [408, 425, 429, 500, 502, 503, 504, 507, 521, 522, 524],
};

// ------------------------------------------------------------
// 4. Payload + Enqueued Job
// ------------------------------------------------------------
export interface JobPayload<TInput = Record<string, unknown>> {
  /** Fair-share key (user-id, tenant-id, anon-session-id, …) */
  userId: string;
  productionId?: string;
  sceneId?: string;
  type: JobType;
  priority: PriorityLevel;
  input: TInput;
  retry?: Partial<RetryPolicy>;
  timeoutMs?: number;
  /** Cancel every job in this group with 1 call (e.g. cancel production render) */
  jobGroup?: string;
}

export interface EnqueuedJob<TInput = unknown, TResult = unknown>
  extends Omit<JobPayload<TInput>, 'retry'> {
  readonly jobId: string;
  readonly createdAtMs: number;
  scheduledAtMs?: number;
  startedAtMs?: number;
  finishedAtMs?: number;
  attempt: number;
  status: JobStatus;
  error?: { readonly code: string; readonly message: string };
  result?: TResult;
  retry: RetryPolicy;
}

// ------------------------------------------------------------
// 5. Adapter Interface (dual-fallback provider pattern — M1)
// ------------------------------------------------------------
export interface AdapterResult<TResult = unknown> {
  success: boolean;
  result?: TResult;
  error?: { code: string; message: string };
  retryable: boolean;
  metrics?: { durationMs: number; peakRssMb?: number };
}

export interface JobAdapter<
  TInput = unknown,
  TResult = unknown,
> {
  type: JobType | readonly JobType[] | 'fallback';
  name: string;
  supports(job: JobPayload<TInput>): boolean;
  execute(
    job: EnqueuedJob<TInput, TResult>,
    signal: AbortSignal,
  ): Promise<AdapterResult<TResult>>;
}

// ------------------------------------------------------------
// 6. Public API (SchedulerHandle) + Events
// ------------------------------------------------------------
export interface SchedulerOptions {
  /** Max concurrency per priority level (0 = unlimited per-level, global cap is sum) */
  maxConcurrencyPerPriority?: Partial<Record<PriorityLevel, number>>;
  defaultConcurrency?: number;
  retryDefaults?: Partial<RetryPolicy>;
  tokenBucket?: {
    capacityPerUser: number;
    refillRatePerSecPerUser: number;
  };
  /** Delay between dispatcher ticks (default = 4ms, ~aligned to browser rAF idle budget) */
  tickMs?: number;
  idleGcIntervalMs?: number;
}

export interface SchedulerStats {
  enqueued: number;
  processed: number;
  failed: number;
  cancelled: number;
  retried: number;
  preEmpted: number;
  missedDeadlines: number;
  countsByPriority: Record<PriorityLevel, number>;
  countsByStatus: Record<JobStatus, number>;
  /** Oldest waiting (queued) job age in ms — or null if queue empty */
  oldestWaitingMs: number | null;
  peakRssMb?: number;
}

export interface SchedulerHandle {
  enqueue<TInput, TResult = unknown>(
    job: JobPayload<TInput>,
  ): Promise<EnqueuedJob<TInput, TResult>>;
  cancel(jobId: string): Promise<boolean>;
  cancelGroup(jobGroup: string): Promise<number>;
  getStatus(jobId: string): Promise<EnqueuedJob | undefined>;
  get snapshot(): SchedulerStats;
  on<K extends keyof SchedulerEvents>(
    event: K,
    cb: (payload: SchedulerEvents[K]) => void | Promise<void>,
  ): () => void;
  registerAdapter(adapter: JobAdapter): void;
  unregisterAdapter(name: string): boolean;
  /** Drain running jobs then free resources; drain=true waits for in-flight */
  stop(drain: boolean): Promise<void>;
}

export interface SchedulerEvents {
  'job:queued': EnqueuedJob;
  'job:start': EnqueuedJob;
  'job:complete': EnqueuedJob;
  'job:fail': EnqueuedJob & { error: NonNullable<EnqueuedJob['error']> };
  'job:retry': EnqueuedJob & { nextAttemptAtMs: number };
  'job:cancel': EnqueuedJob;
  'adapter:fallback': {
    jobId: string;
    primary: string;
    fallback: string;
  };
  'scheduler:idle': { consecutiveIdleTicks: number };
  'scheduler:tick-overrun': { tickMs: number; budgetExceededMs: number };
}

// ------------------------------------------------------------
// 7. Heap Node (binary min-heap key) — FIFO within same priority
// ------------------------------------------------------------
/**
 * Tuple sort order:
 * 1. priority (lower = sooner): URGENT(0) < HIGH(1) < BG(2) < IDLE(3)
 * 2. createdAtMs (older = sooner): FIFO starvation prevention inside same priority
 */
export type HeapKey = readonly [priority: PriorityLevel, createdAtMs: number];

export interface HeapNode<TInput = unknown, TResult = unknown> {
  readonly key: HeapKey;
  readonly jobId: string;
  readonly job: EnqueuedJob<TInput, TResult>;
}
