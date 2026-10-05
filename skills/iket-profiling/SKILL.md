---
name: iket-profiling
description: Profile inside a CuTe DSL (cutlass.cute) kernel with IKET (In-Kernel Event Tracing) and the run-iket profiler. Use when developing or tuning a CuTe DSL kernel and you need per-warp timelines of kernel phases (setup, mainloop, epilogue), producer/consumer (TMA/MMA) activity, or time spent in pipeline and mbarrier waits. Not for kernel launch latency, inter-kernel gaps, kernel overlap, or whole-application timelines (use Nsight Systems), and not in the same run as Nsight Compute, Nsight Systems, or other CUPTI tools.
license: Apache-2.0
compatibility: Requires an nvidia-cutlass-dsl installation that includes run-iket, and an SM90, SM100, SM103, SM110, or SM120 NVIDIA GPU.
---

# IKET profiling for CuTe DSL kernels

IKET lets CuTe DSL kernels emit named markers and ranges from device code, like NVTX but inside the kernel. The `run-iket` profiler, released with `nvidia-cutlass-dsl`, collects them into a Perfetto trace (`*.pftrace`, open at https://ui.perfetto.dev/) and JSON. IKET is experimental: the API, output format, and overhead may change. The source of truth is the official guide: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html

## Workflow

1. Check the profiler is available: `run-iket --help`.
2. Instrument the `@cute.kernel` function. Calls in host-side code (`@cute.jit` functions, launch wrappers) do not emit in-kernel events.

   ```python
   import cutlass
   import cutlass.cute as cute

   @cute.kernel
   def kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
       bidx, _, _ = cute.arch.block_idx()
       cute.experimental.iket.mark("kernel_start", bidx)

       load_token = cute.experimental.iket.range_start("load")
       # Load data from gA and gB.
       cute.experimental.iket.range_end(load_token)

       cute.experimental.iket.range_push("compute")
       # Compute and store results.
       cute.experimental.iket.range_pop()
   ```

3. Run the workload under `run-iket`. The workload command goes after `--`. The kernel must be JIT-compiled inside this process; a kernel that was already compiled and is reused gets no instrumentation.

   ```bash
   run-iket --output-dir ./iket_output --clobber profile --postprocess all -- python my_kernel.py
   ```

   `--postprocess perfetto|json|all` selects the output. For large grids, add `--enabled-cluster X,Y,Z` to dump only one cluster (a non-cluster kernel counts as a 1x1x1 cluster, so this selects one thread block). Use `--enabled-cluster-config FILE.json` to choose per kernel: a JSON object mapping a substring of the mangled kernel name to `[x, y, z]`, `null`, or `"all"`.

4. Open the `*.pftrace` in Perfetto. Pan and zoom with W/A/S/D. Tracks are grouped by GPU location, then CTA and warp. `WarpLifeTime` tracks are automatic; each token-range name has its own track; push/pop ranges are on `StackedRanges`; markers are on `Marker`.
5. Start with a few coarse ranges (one whole-kernel `range_start`/`range_end`, plus setup, mainloop, epilogue), then add detail only where the trace shows it is needed.

Complete example in the CUTLASS repo: `examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py`.

## API

All calls are in `cutlass.cute.experimental.iket` and belong inside `@cute.kernel` code.

| Call | Meaning |
|---|---|
| `mark(name[, payload])` | Point event. |
| `range_push(name[, payload])`, `range_pop()` | Stack-based range. `range_pop` takes no name and closes the most recent push (LIFO). |
| `range_start(name[, payload]) -> token`, `range_end(token[, payload])` | Token-based range. Use when the range ends at a later sync point, crosses an iteration, or has several mutually exclusive close sites. |
| `sentinel_token(name)` | Token that emits no event. `range_end` on it is a no-op. Use for cross-iteration ranges where `range_end` comes before `range_start` in source order. |

Payloads are Python bool, int, or float literals, or CuTe DSL numeric or index scalars. Do not use tensors, tuples, or other aggregates. Int and float literals become 32-bit; use `cutlass.Int64(...)` or `cutlass.Float64(...)` for 64-bit, which costs more. A token range with a start payload must have an end payload of the same type, and vice versa.

Events are warp-level. If threads in a warp evaluate a payload differently, the first active thread's value is recorded. To record a specific thread's value, guard the call (for example `if tidx == 0:`) and guard both endpoints of a range the same way.

## Rules

Breaking these can give wrong timelines, undefined results, or a failed postprocess.

- Each dynamic `range_push` has exactly one `range_pop` on every participating warp execution path, in LIFO order, with the same nesting on every path.
- Each dynamic non-sentinel `range_start` is closed by `range_end` once on every executed path. Closing the same token in mutually exclusive branches is fine.
- Never start a range in one thread-divergent branch and end it in another. Never return or exit early between paired endpoints.
- In warp-specialized code, put both endpoints inside the same role guard (`if warp_idx == tma_warp_id:`). Prefixing names by role (`tma_`, `mma_`) makes the JSON easy to filter.
- Names are at most 32 characters. Reuse one name for a recurring loop phase. More than 30 unique marker and range names increases overhead.
- Do not put IKET range calls inside `cutlass.range(..., prefetch_stages=...)`.
- Avoid high-frequency events in innermost, unrolled loops. IKET calls act partly as code-motion barriers and can change compiler scheduling.

## Patterns

Async work (TMA, cp.async, MMA): a range around the issue point measures issue time only. Completion is observed at the wait, so wrap the wait:

```python
cute.experimental.iket.range_push("ab_wait")
ab_full = ab_consumer.wait_and_advance()
cute.experimental.iket.range_pop()
```

Cross-iteration range, where tile N's range closes at iteration N + 1's wait:

```python
iter_token = cute.experimental.iket.sentinel_token("mma_k_tile")
for k_tile in cutlass.range(k_tile_count):
    ab_full = ab_consumer.wait_and_advance()
    if k_tile > 0:
        cute.experimental.iket.range_end(iter_token)
    iter_token = cute.experimental.iket.range_start("mma_k_tile")
    # Work for this tile.
    ab_full.release()
if k_tile_count > 0:
    # Final drain or synchronization for the last tile.
    cute.experimental.iket.range_end(iter_token)
```

If start and end are in the same iteration, a `range_push("k_tile", k_tile)` / `range_pop()` pair inside the loop is simpler.

## Building with IKET outside run-iket

IKET calls are stripped by default. To keep them in a build without `run-iket`, set `CUTE_DSL_COMPILER_OPT=iket` for all JIT compilations in the process, or pass `options="iket"` to `cute.compile(host_function, *args, options="iket")`.

## JSON output

The postprocessed file is usually `iket_pid_<pid>.trace.json`; its schema is experimental. Names and warp locations are stored once and referenced by index:

- `stringTable[]`: names. Ranges use `rangeNameIdx`; markers use `markerNameIdx`.
- `locationTable[]`: warp locations with `smId`, `tpcId`, `gpcId`, `ctaId`, `warpId`.
- `launches[]`: one per profiled launch, with `contextId`, `gridId`, `kernelName`, `gridDim*`, `blockDim*`, and:
  - `ranges[]`: `startTs`, `endTs` (duration is `endTs - startTs`), `warpLocIdxs[]`.
  - `markers[]`: `timestamp`, `locIdx`, and `payloadType`/`payloadVal` when a payload exists.
  - `warpLifetimes[]`: `startTs`, `endTs`, `locIdx`.
- `graphLaunches`: CUDA Graph launches, keyed by strings such as `graph_exec_1:0`, each an array shaped like `launches[]`.

Timestamps are trace-local: compare differences within one trace, never absolute values across traces.

## Limitations

- In-kernel timing only. For launch latency, inter-kernel gaps, kernel overlap, or CPU/GPU scheduling, use Nsight Systems in a separate run.
- Cannot run at the same time as Nsight Compute, Nsight Systems, or other CUPTI-based tools.
- Timer granularity is 32 ns; a very short range can have identical start and end timestamps.
- The whole workload is profiled; there is no capture window. Keep it small with few instrumented launches, or it may run very slowly or run out of memory.
- Buffers are sized by an earlier pass and assume a stable per-warp event count. Data-dependent event rates can corrupt the trace or cause an illegal memory access.
- Multi-process or multi-GPU runs may produce separate traces that are not aligned to one timeline.
- Payloads, and especially 64-bit payloads, add overhead and trace volume.
- To measure instrumentation overhead, compare trace-reported kernel durations in Perfetto between a run with one instrumented event site and a run with the full set. Do not use host wall-clock time.

## Troubleshooting

- Empty or missing trace: the workload was not run under `run-iket`, or the kernel was not JIT-compiled in the profiled process. Without `run-iket`, check `CUTE_DSL_COMPILER_OPT=iket` or `options="iket"`.
- Expected events missing: the calls are not in `@cute.kernel` code, are unreachable on the profiled path, or have endpoints split across divergent branches. If the application caches compiled kernels, compile inside the profiled process or clear that cache.
- Trace too large, or profiling fails: reduce instrumented launches, event frequency, or payloads; use `--enabled-cluster`. If a warp can legitimately emit more events than detected, pass `--max-ts-cnt-per-warp <N>` with N above the largest per-warp event count in one launch.
- Unexpected timing around async work: move the range endpoint to the wait or sync point that observes completion.
- To check that IKET operations were emitted, inspect the IR with `CUTE_DSL_KEEP=ir` or `CUTE_DSL_PRINT_IR=1`.
