# Contributing to MovieSer Core

Thanks for considering contributing! 🙏 This document explains how we work.

---

## Project phases & contribution rules

MovieSer Core is being built under two formal research programmes:
- **WBSO** (RVO The Netherlands) — 500 h research (hypothesis-driven)
- **NLnet NGI0 Commons** — 400 h + € 20 000 grant

Because of this, we split the project into phases and **strictly separate
implementation from the WBSO research loop**. Please respect these
contribution windows:

### Phase 1 — Research (Milestones M1, M2, M3) — WBSO running
> Until M4 (± Week 16)

**Accepting:**
- ✅ **Bug reports** — open a GitHub Issue with reproduction
- ✅ **Security disclosures** — email security@solhei-solutions.nl (PGP on request)
- ✅ **Documentation / typo fixes** in existing READMEs

**NOT yet accepting:**
- ❌ Feature PRs — deferred to M4. The WBSO research plan is fixed per loop.
- ❌ Refactors, style changes, lint-only PRs — these conflict with the
  time-stamped WBSO research logbook (RVO inspects the commit history)
- ❌ Breaking API changes — API is frozen for the 4 milestones

### Phase 2 — Public v1.0.0 (Milestone M4) — Post research
> After M4 delivery + NLnet acceptance

Opening for:
- ✅ Bug fixes, performance improvements
- ✅ Integration PRs (React / Vue / Svelte adapter packages)
- ✅ OPFS / IndexedDB backend enhancements
- ✅ Benchmark dataset expansions

---

## Bug reports — what to include

When opening an Issue, include:

1. Package + exact version: `@movieser/scheduler@0.1.3`
2. Browser / OS / device: "Chrome 128, Windows 11, 8 GB RAM, i5-8250U"
3. Minimal reproduction: a StackBlitz or code snippet
4. Expected vs. actual behavior
5. If a performance bug: Chrome Tracing export or Firefox Profiler link

For **security** issues: do NOT create a public issue. Email
security@solhei-solutions.nl instead. We respond within 72 business hours.

---

## Commit format & DCO sign-off

We do NOT use a CLA. We use the **Developer Certificate of Origin (DCO)**.

Before you commit, add a `Signed-off-by` trailer using your real name and email:

```bash
git commit -s -m "fix(scheduler): pre-empt P1 when P0 arrives mid slice
Fixes: #123

Signed-off-by: Your Name <your.name@example.com>"
```

Or use `git commit -s` (the `-s` flag appends the line automatically) using
`git config user.name` and `git config user.email` set to your identity.

Commits without DCO sign-off **cannot be merged**. This is a hard requirement
from the NLnet Commons fund auditing chain.

---

## Code style

We follow a very small set of rules. Run `npm test` before opening a PR:

- TypeScript strict `strict: true` plus `exactOptionalPropertyTypes` and
  `noUncheckedIndexedAccess` (set per package `tsconfig.json`)
- Prefer zero dependencies — adding a dep requires a justification in the PR
  description
- Placeholder files stay placeholder until WBSO logboek documents the
  H₀/H₁ → experiment → measurement → conclusion that justifies code changes
- Test coverage target for exported symbols: ≥ 95 % (Vitest `coverage`)

---

## Release cadence

| Event | Channel |
| --- | --- |
| Every milestone end (M1, M2, M3) | `-canary` npm release + changelog |
| M4 | `v1.0.0` public stable + all 3 packages |
| After M4 | Minor: monthly-ish; Patch: on demand |

---

## Governance (post-M4)

After v1.0.0 we will form a **Special Interest Group (SIG): Browser
Performance & Inclusive Editing Tools**. SIG meetings (initially quarterly)
decide on roadmap, and maintainers (Solhei + 2-3 community volunteers) run
day-to-day merges.

For now: all merge decisions are made by Solhei Solutions as lead maintainer
and principal investigator on the WBSO / NLnet projects.
