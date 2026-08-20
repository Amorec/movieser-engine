// ============================================================
// Silent Persistence Worker Module
// ============================================================
// Runs inside a Web Worker. Receives SAB-backed buffers (SharedArrayBuffer)
// + Atomics for zero-copy signalling.
//
// Responsibilities:
//  - Execute structuredClone / Serialization OFF main thread
//  - Open IndexedDB / OPFS handles
//  - Batch commits to WAL (5+ mutations per transaction minimum)
//  - Fire sync signal back when committed (no UI, no notificaties)
//
// Will be implemented during NLnet M3 / WBSO Onderzoekslus III.
// File exists so TypeScript bundler can inline worker.
