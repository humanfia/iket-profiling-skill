# iket-profiling-skill

> **This is NOT new work.** This repo is just a condensed summary of NVIDIA's official CUTLASS documentation on IKET profiling for CuTe DSL kernels, repackaged as an agent skill. All technical content comes from the official docs — treat them as the source of truth:
>
> - Official guide: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/iket_profiling.html
> - Official example: `examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py` in the [CUTLASS repo](https://github.com/NVIDIA/cutlass)

## What it is

A single [`SKILL.md`](SKILL.md) in the open [Agent Skills](https://agentskills.io) format (YAML frontmatter + markdown). It activates when an agent is developing a CuTe DSL (`cutlass.cute`) kernel and tuning its performance, and covers: the IKET API (`mark` / `range_push`·`range_pop` / `range_start`·`range_end` / `sentinel_token`), the `run-iket` profiler workflow, instrumentation rules, common patterns (async issue vs. wait timing, cross-iteration ranges), output formats (Perfetto / JSON), limitations, and troubleshooting — all per the official guide.

## Install

**Claude Code** (personal, all projects):

```bash
git clone git@github.com:humanfia/iket-profiling-skill.git ~/.claude/skills/iket-profiling
```

or into a project: `.claude/skills/iket-profiling/`.

**Codex CLI** (versions supporting the Agent Skills standard):

```bash
git clone git@github.com:humanfia/iket-profiling-skill.git ~/.codex/skills/iket-profiling
```

If your agent doesn't support skills, just point it at `SKILL.md` (e.g. reference it from `AGENTS.md`).

## Attribution

IKET, CuTe DSL, and CUTLASS are NVIDIA projects. This summary was last checked against the docs in August 2026; IKET is experimental and its API/output may change — always defer to the official documentation.
