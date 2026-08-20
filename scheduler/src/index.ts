// ============================================================
// @movieser/scheduler — Placeholder (to be implemented in WBSO Onderzoekslus II)
// ============================================================
//
// This stub is intentionally minimal. The real implementation is part of
// WBSO Onderzoekslus II (Milestone 1 NLnet) and will contain:
//  - PriorityQueue with 3 tiers: P0 (user-blocking), P1 (render), P2 (idle)
//  - requestIdleCallback integration with deadline pre-emption
//  - Zero dependencies, framework-agnostic
//  - Target: 99.7% of events <=1ms, 0% missed input events
//
// Do NOT implement features here outside the WBSO research loop.
// Research order: H0/H1 -> Experiment -> Measurement -> Conclusion -> Code

export type PriorityLevel = 0 | 1 | 2;

export interface QueueOptions {
  /** If true, logs scheduling decisions (debug only, < 0.1% overhead) */
  debug?: boolean;
  /** Fallback for SSR / old browsers: setTimeout with this delay (ms) */
  idleFallbackDelayMs?: number;
}

export interface SchedulerStats {
  enqueued: number;
  processed: number;
  preEmpted: number;
  missedDeadlines: number;
}

export const Priority = {
  P0_USER_BLOCKING: 0 as const,
  P1_RENDER_DIFF: 1 as const,
  P2_IDLE_ONLY: 2 as const,
} as const;

export class PriorityQueue {
  constructor(_options: QueueOptions = {}) {
    // TODO Milestone 1 (WBSO Onderzoekslus II)
    // Implement queue + requestIdleCallback dispatcher
    // Do NOT build before Hypothesis H0/H1 are documented in WBSO logbook
  }

  enqueue<T>(
    _priority: PriorityLevel,
    _task: () => T,
    _opts?: { deadlineMs?: number; signal?: AbortSignal },
  ): Promise<T> {
    // PLACEHOLDER: direct execution, no scheduling
    // Will be replaced in M1 research implementation
    return Promise.resolve(_task());
  }

  clear(_priority?: PriorityLevel): void {
    // Placeholder
  }

  get stats(): SchedulerStats {
    return { enqueued: 0, processed: 0, preEmpted: 0, missedDeadlines: 0 };
  }

  dispose(): void {
    // Cleanup listeners, clear queues
  }
}
