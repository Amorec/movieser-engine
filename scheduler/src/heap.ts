// ============================================================
// @movieser/scheduler — Binary Min-Heap (Priority Queue internals)
// ============================================================
// WBSO Milestone 1 | NLnet M1
// Performance: O(log n) insert / extractMin / decreaseKey (best-effort)
// Key = [priority, createdAtMs] → tuple (FIFO starvation prevention)
// Zero dependencies, framework agnostic, Node + Browser
// ============================================================

import type { HeapKey, HeapNode } from './types.js';

/**
 * Tuple comparison for the heap key. Returns -ve if a < b, +ve if a > b, 0 if equal.
 * Priority first (numeric). createdAtMs second (older first → FIFO inside same priority).
 */
export function compareHeapKey(a: HeapKey, b: HeapKey): number {
  const p = a[0] - b[0];
  if (p !== 0) return p;
  const t = a[1] - b[1];
  if (t !== 0) return t;
  return 0;
}

/**
 * Binary min-heap over HeapKey = [priority, createdAtMs].
 * Backed by a dynamic array (0-indexed with 0 sentinel for easier parent/child math).
 *
 * Indices (when sentinel at [0]):
 *   parent(i) = Math.floor(i / 2)
 *   left(i)   = 2*i
 *   right(i)  = 2*i + 1
 */
export class BinaryHeap<TInput = unknown, TResult = unknown> {
  private readonly _arr: Array<HeapNode<TInput, TResult>>;

  constructor() {
    // sentinel slot at index 0 — kept empty, never swapped
    this._arr = [null as unknown as HeapNode<TInput, TResult>];
  }

  get size(): number {
    return this._arr.length - 1;
  }

  /**
   * Insert a new node. O(log n).
   */
  insert(node: HeapNode<TInput, TResult>): void {
    this._arr.push(node);
    this._bubbleUp(this._arr.length - 1);
  }

  /**
   * Peek top without removing. O(1). Returns null if empty.
   */
  peek(): HeapNode<TInput, TResult> | null {
    if (this.size === 0) return null;
    return this._arr[1] as HeapNode<TInput, TResult>;
  }

  /**
   * Extract and return the min-key node. O(log n). Returns null if empty.
   */
  extractMin(): HeapNode<TInput, TResult> | null {
    if (this.size === 0) return null;
    const top = this._arr[1] as HeapNode<TInput, TResult>;
    const last = this._arr.pop() as HeapNode<TInput, TResult>;
    if (this.size > 0) {
      this._arr[1] = last;
      this._bubbleDown(1);
    }
    return top;
  }

  /**
   * Best-effort decreaseKey: re-insert filtered by jobId if priority/createdAt
   * of a node became smaller. Since our key is immutable (priority + createdAt
   * never shrink for a live queued job today), this is lazy in M1 v0.1.
   *
   * Returns true if any structural change happened.
   */
  rebuildFiltered(
    predicate: (node: HeapNode<TInput, TResult>) => boolean,
  ): number {
    const kept = this._arr
      .slice(1)
      .filter((n) => predicate(n as HeapNode<TInput, TResult>))
      .sort((a, b) => compareHeapKey(a.key, b.key));
    this._arr.length = 1;
    for (const n of kept) this._arr.push(n as HeapNode<TInput, TResult>);
    return kept.length;
  }

  /**
   * Iterate (read-only) queued nodes in insertion-array order (not key order).
   * Cheap O(n), no copy. Used by stats/snapshot only.
   */
  forEach(cb: (node: HeapNode<TInput, TResult>) => void): void {
    for (let i = 1; i < this._arr.length; i++) {
      cb(this._arr[i] as HeapNode<TInput, TResult>);
    }
  }

  clear(): void {
    this._arr.length = 1;
  }

  // ------------------------------------------------------------
  // Private helpers
  // ------------------------------------------------------------

  private _bubbleUp(i: number): void {
    const arr = this._arr;
    while (i > 1) {
      const pi = Math.floor(i / 2);
      const parent = arr[pi] as HeapNode<TInput, TResult>;
      const child = arr[i] as HeapNode<TInput, TResult>;
      if (compareHeapKey(child.key, parent.key) >= 0) break;
      arr[pi] = child;
      arr[i] = parent;
      i = pi;
    }
  }

  private _bubbleDown(i: number): void {
    const arr = this._arr;
    const n = arr.length;
    while (true) {
      const li = 2 * i;
      const ri = li + 1;
      let smallest = i;
      if (
        li < n &&
        compareHeapKey(
          (arr[li] as HeapNode<TInput, TResult>).key,
          (arr[smallest] as HeapNode<TInput, TResult>).key,
        ) < 0
      ) {
        smallest = li;
      }
      if (
        ri < n &&
        compareHeapKey(
          (arr[ri] as HeapNode<TInput, TResult>).key,
          (arr[smallest] as HeapNode<TInput, TResult>).key,
        ) < 0
      ) {
        smallest = ri;
      }
      if (smallest === i) break;
      const tmp = arr[i] as HeapNode<TInput, TResult>;
      arr[i] = arr[smallest] as HeapNode<TInput, TResult>;
      arr[smallest] = tmp;
      i = smallest;
    }
  }
}
