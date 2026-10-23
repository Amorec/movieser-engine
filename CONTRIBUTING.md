# Contributing to MovieSer Core

Thanks for considering contributing! 🙏 This document explains how we work.

---

## What we accept

MovieSer Core is an early-stage library suite, and the modules stay
**unstable until v1.0.0**. The public API of the three core packages is
therefore **frozen** — decisions on interfaces are the maintainer's. Anything
that does not change the API is welcome.

**Welcome:**
- ✅ **Bug reports** — open a GitHub Issue with a reproduction
- ✅ **Security disclosures** — email security@solhei-solutions.nl (PGP on request)
- ✅ **Bug fixes** and correctness improvements
- ✅ **Documentation** — READMEs, guides, examples, typos
- ✅ **Test coverage** for existing behaviour
- ✅ **Adapter packages** — React / Vue / Svelte bindings, shipped as separate packages
- ✅ **Benchmark datasets** and reproducibility reports

**Not yet:**
- ❌ Changes to the public API of `scheduler`, `ctm-engine` or `silent-persistence`
  (frozen until v1.0.0)
- ❌ New runtime dependencies in the core packages — open an issue first
- ❌ Project-wide refactors or formatting-only PRs — hard to review, no value

Not sure whether your idea fits? Open an issue before writing code.

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

Commits without DCO sign-off **cannot be merged**.

---

## Code style

We follow a very small set of rules. Run `npm test` before opening a PR:

- TypeScript strict `strict: true` plus `exactOptionalPropertyTypes` and
  `noUncheckedIndexedAccess` (set per package `tsconfig.json`)
- Prefer zero dependencies — adding a dep requires a justification in the PR
  description
- Placeholder files stay placeholder until the research logbook documents the
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

For now: all merge decisions are made by Solhei Solutions as lead maintainer.
