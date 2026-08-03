---
name: iket-profiling
description: Profile inside a CuTe DSL kernel with IKET (In-Kernel Event Tracing) and the run-iket profiler. Use when developing a CuTe DSL (cutlass.cute) kernel and tuning its performance — to get per-warp timelines of kernel phases (setup/mainloop/epilogue), see producer/consumer (TMA/MMA) overlap, and find where warps stall on pipeline or mbarrier waits. Triggers include "profile the cute kernel", "where do warps wait/stall", "in-kernel timeline", "IKET", "run-iket", "调优 cute kernel", "看 kernel 内部时间线". NOT for kernel launch latency, kernel overlap, or whole-app timelines (use Nsight Systems), and it cannot run together with ncu/nsys.
---

# IKET Profiling for CuTe DSL Kernels

IKET (In-Kernel Event Tracing) lets CuTe DSL kernels emit named markers and ranges from device code (like device-side NVTX). The `run-iket` profiler (ships with the `nvidia-cutlass-dsl` package; check with `run-iket --help`) collects them into a Perfetto timeline (`.pftrace`, view at https://ui.perfetto.dev/) and machine-readable JSON. Supports Hopper and newer: SM90, SM100, SM103, SM110, SM120. Experimental — API and output format may change.

## Workflow

1. **Instrument** the `@cute.kernel` function (host-side Python calls emit nothing):

```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel(...):
    bidx, _, _ = cute.arch.block_idx()
    cute.experimental.iket.mark("kernel_start", bidx)     # point marker (+ optional payload)

    load_token = cute.experimental.iket.range_start("load")  # token-based range
    # ... loads ...
    cute.experimental.iket.range_end(load_token)

    cute.experimental.iket.range_push("compute")             # stack-based range (LIFO)
    # ... compute/store ...
    cute.experimental.iket.range_pop()
```

2. **Profile** (workload command goes after `--`; the kernel must JIT-compile inside this run — cached compiled kernels get no instrumentation):

```bash
run-iket --output-dir ./iket_output --clobber profile --postprocess all -- python my_kernel.py
```

`--postprocess perfetto|json|all` selects output. Open the `.pftrace` in Perfetto UI (pan/zoom with W/A/S/D; tracks are grouped by SM/CTA/warp: `WarpLifeTime`, per-name token-range tracks, `StackedRanges`, `Marker`).

3. **Iterate**: start with a few coarse phase ranges (setup / mainloop / epilogue + one whole-kernel `range_start` at entry, `range_end` at exit), then drill into waits and per-iteration detail only where needed.

## API (`cutlass.cute.experimental.iket`, inside `@cute.kernel` only)

| API | Purpose |
|---|---|
| `mark(name[, payload])` | Point event |
| `range_push(name[, payload])` / `range_pop()` | Stack-based range; pop closes the most recent push (LIFO, no name) |
| `range_start(name[, payload]) -> token` / `range_end(token[, payload])` | Token-based range; use when the end is at a later sync point or in multiple mutually exclusive close sites |
| `sentinel_token(name)` | Placeholder token (emits no event) for cross-iteration ranges where `range_end` appears before `range_start` in source order |

Payloads: Python bool/int/float literals or CuTe DSL scalars only (no tensors/tuples). Int/float literals become 32-bit; use `cutlass.Int64(...)` / `cutlass.Float64(...)` for 64-bit (higher overhead). Events are **warp-level**: a divergent payload records the first active thread's value — guard with e.g. `if tidx == 0:` (guard both endpoints consistently). Token ranges must use matching payload presence and type at start and end.

## Instrumentation rules (violations can corrupt or fail profiling)

- Every dynamic `range_push` has exactly one matching `range_pop` per participating warp execution path (LIFO); every non-sentinel `range_start` is closed once per executed path (closing the same token in mutually exclusive branches is fine).
- Never start a range in one thread-divergent branch and end it in another; no early exit between paired endpoints. In warp-specialized code, put both endpoints inside the same role guard (`if warp_idx == tma_warp_id:` …) and prefix names by role (`tma_`, `mma_`) for easy filtering.
- Names ≤ 32 chars; reuse the same name for a recurring loop phase. More than 30 **unique** names increases overhead.
- No IKET range ops inside `cutlass.range(..., prefetch_stages=...)` — unsupported.
- Avoid high-frequency events in innermost (unrolled) hot loops: IKET calls can act as code-motion barriers and change compiler scheduling.

## Key patterns

**Issue vs. completion of async work** (TMA, cp.async, MMA): a range right around the issue point measures issue time only. To measure completion/wait, wrap the wait:

```python
cute.experimental.iket.range_push("ab_wait")
ab_full = ab_consumer.wait_and_advance()   # pipeline/mbarrier wait
cute.experimental.iket.range_pop()
```

**Cross-iteration range** (tile N's range closes at iteration N+1's wait boundary):

```python
iter_token = cute.experimental.iket.sentinel_token("mma_k_tile")
for k_tile in cutlass.range(k_tile_count):
    ab_full = ab_consumer.wait_and_advance()
    if k_tile > 0:
        cute.experimental.iket.range_end(iter_token)   # close previous tile
    iter_token = cute.experimental.iket.range_start("mma_k_tile")
    # ... work ...
if k_tile_count > 0:
    cute.experimental.iket.range_end(iter_token)       # after final drain
```

Full example: CUTLASS repo `examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py`.

## Enabling IKET without run-iket

IKET calls are **stripped by default**. To keep instrumentation in a normal build: `export CUTE_DSL_COMPILER_OPT=iket`, or per-compilation `cute.compile(host_fn, *args, options="iket")`.

## JSON output (script-based analysis)

`launches[]` → per profiled launch: `gridId`, `kernelName`; `ranges[]` (`rangeName`, `startTs`, `endTs`, `warpLocs[]` with `smId`/`tpcId`/`gpcId`/`ctaId`/`warpId`); `markers[]` (`markerName`, `timestamp`, `location`, `payloadType`/`payloadVal`); `warpLifetimes[]`. Duration = `endTs - startTs`. Timestamps are trace-local — never compare across traces.

## Limitations & pitfalls

- **Cannot run concurrently with Nsight Compute / Nsight Systems / any CUPTI tool** — separate runs.
- In-kernel timing only: don't use IKET for launch latency, inter-kernel gaps, overlap, or CPU/GPU scheduling — use Nsight Systems.
- Timer granularity is 32 ns; very short ranges may show identical start/end.
- No capture window: the whole workload is profiled. Keep it small (few instrumented launches), else it runs slowly or OOMs; records are per warp across many CTAs.
- Buffer sizing uses two passes and assumes a stable per-warp event count; data-dependent event rates can corrupt the trace or cause illegal memory access. If a warp can emit more events than auto-detected, pass `--max-ts-cnt-per-warp <N>`.
- Multi-process / multi-GPU runs produce separate traces NOT aligned to one timeline.
- To measure instrumentation overhead: compare **trace-reported kernel durations** between a minimal (one event) and a full instrumentation run — not host wall-clock.

## Troubleshooting

- **Empty/missing trace**: workload not launched under `run-iket`, or the kernel wasn't JIT-compiled in the profiled process (clear compiled-kernel caches), or missing `CUTE_DSL_COMPILER_OPT=iket` / `options="iket"` when running standalone.
- **Missing events**: calls not inside `@cute.kernel`, unreachable path, or paired endpoints split across divergent branches.
- **Huge trace / failure**: fewer launches, fewer events, fewer payloads; raise `--max-ts-cnt-per-warp`.
- **Odd async timing**: move the range endpoint to the wait/sync point that observes completion.
- **Inspect IR**: `CUTE_DSL_KEEP=ir` or `CUTE_DSL_PRINT_IR=1` to verify iket ops were emitted.

Official doc: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/iket_profiling.html
