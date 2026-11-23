# Inkling on M3 Pro: results and configuration

Status: completed on 23 November 2026. Exact repeated-prompt caching passed the latency, output and memory gates and is available as an explicit option. No new decode-throughput optimization met the 5% balanced improvement gate.

For the combined account of both chats and an explanation of every optimization file, start with the [complete work guide](inkling-m3-complete-guide.md) and [file map](inkling-m3-file-map.md).

## Fixed reference

Hardware: Apple M3 Pro, 18-core GPU, 18 GiB unified memory. MLX 0.32.4 and mlx-lm 0.32.2. Target: cached Inkling-2-Mini-LookAtMyQuant-8.9GB revision `b5fe3bee6a8dc6dd50bb00c2dd17d2e259365a85`; drafter revision `adac74d1cf00ed68a19a77233a62bd6dcc6c2607`. All work used cached files with `HF_HUB_OFFLINE=1`. Target weights, quantization, FP16 KV, full-target verification and reasoning settings are preserved. Rendered prompts use `reasoning_effort=none`, matching the original CLI command without `--thinking`.

The new [complete 256-token baseline](../results/inkling_m3/baseline-balanced-256-complete.json) contains 90 measured runs: nine prompts, five repetitions, serial and existing drafted modes. Both modes were stable within their own repetitions. Their greedy outputs differ on some prompts, as already observed before this continuation.

| Family | Existing serial median | Existing drafted median |
|---|---:|---:|
| Math | 28.05 tok/s | 41.72 tok/s |
| Coding | 27.59 tok/s | 38.16 tok/s |
| Chat | 26.70 tok/s | 26.21 tok/s |

These medians describe the existing implementations; they are not a new speedup. The interrupted legacy `baseline-balanced-256.json` remains preserved separately.

## Accepted exact retry cache

An `ExactPromptCache` saves the original complete prefill, including the first-token result, final hidden taps, all target cache state and the drafter context before the final returned taps are consumed. It reuses only an identical rendered prompt with compatible execution settings. Extensions and edited histories get a fresh original prefill. This avoids the split-prefill numerical differences found with the existing prefix cache.

The [balanced validation](../results/inkling_m3/exact-cache-balanced-analysis.json) passed all 90 measured runs, including exact token IDs and stopping. It records snapshot fills and separate uncached memory references, so a resident snapshot cannot hide its memory cost.

| Family | Typical uncached first token | Typical cached first token | Family decode ratio |
|---|---:|---:|---:|
| Math | 264.87 ms | 3.09 ms | 0.9991x |
| Coding | 230.59 ms | 3.32 ms | 1.0009x |
| Chat | 220.18 ms | 3.16 ms | 0.9937x |

The geometric family improvements in first-token latency were 81.47x / 73.05x / 71.93x. End-to-end response gains on these short, up-to-256-token cases were about 0.8–7%, because decoding still dominates response time. Decode throughput was effectively unchanged. Balanced peak active MLX memory was also effectively unchanged, with a largest measured increase of 384 bytes including fills. Retention is bounded by both entry count and the sum of retained array sizes; measured peak memory is checked separately.

Current public serving option:

```sh
lamq serve <cached-model-path> --drafter <cached-drafter-path> \
  --reasoning-effort none --prompt-cache 1 --prompt-cache-mode exact
```

The retained-array budget defaults to 512 MiB and can be set with `--prompt-cache-mib`. The server defaults remain unchanged. Omit `--prompt-cache` or set it to zero to disable snapshots. Python users can import `ExactPromptCache` from `runtime_mlx.inkling` and pass it to `stream_dspark` or `stream_serial`. Clear snapshots when replacing model state in place. Sampling occurs outside the cached prefill; cached output is not a saved random draw.

For larger prompts on the tested M3 Pro, use the validated optional configuration:

```sh
HF_HUB_OFFLINE=1 lamq serve <cached-model-path> --drafter <cached-drafter-path> \
  --reasoning-effort none --prompt-cache 1 --prompt-cache-mode exact \
  --prompt-cache-mib 1536 --prompt-cache-low-memory
```

Low-memory mode recomputes identical FP16 prefill loop weights and caps unused allocator cache at 256 MiB during generation, honoring a tighter existing limit. It clears unused buffers and restores the previous allocator and wired limits on completion, cancellation and cleanup error. This policy is effective only on fast Apple M3 Pro execution. Python configuration is `ExactPromptCache(max_entries=1, max_bytes=1536 * 1024**2, low_memory=True)`.

These flags apply to MLX serving. Reuse remains restricted to the existing trusted single-user loopback configuration; authentication or a network bind disables it. A snapshot exceeding the retained-array budget is not stored. The initial request and any extension or edit still run a full prefill. The measured gains therefore apply to identical retries while their snapshot remains resident.

## Broader validation and long prompts

| Study | Coverage | Result |
|---|---|---|
| [Held-out](../results/inkling_m3/exact-cache-heldout-analysis.json) | Three prompts, five pairs, 256-token cap | All 30 runs matched; cache gate passed |
| [Histories](../results/inkling_m3/exact-cache-histories-analysis.json) | Nine original, extended and edited cases | All 18 runs matched; extensions and edits used fresh prefill |
| [Sampling](../results/inkling_m3/exact-cache-sampled-seven-analysis.json) | Seven seeded pairs per family, temperature 0.7, 64 tokens | All 42 runs matched; decode ratios 1.0014 / 0.9896 / 1.0051 |
| [Long prompts](../results/inkling_m3/exact-cache-long-analysis.json) | Five pairs each at 2K, 8K and 16K, 128 tokens | All 30 runs matched; memory bounds passed; initial 2K timing required a rerun |

The final long-prompt measurements for the optional low-memory configuration are:

| Context | Uncached first token | Cached first token | Decode ratio | Reference condition |
|---|---:|---:|---:|---|
| [2K](../results/inkling_m3/exact-cache-2k-seven-analysis.json) | 4.923 s | 27.454 ms | 1.0025x | Seven pairs against original allocator settings |
| [8K](../results/inkling_m3/exact-cache-8k-original-reference-analysis.json) | 20.853 s | 42.769 ms | 1.0061x | Five pairs against original allocator settings |
| [16K](../results/inkling_m3/exact-cache-long-analysis.json) | 43.134 s | 36.336 ms | 0.9944x | Five pairs; both arms used a 256 MiB allocator cap |

The 16K result compares caching under a common allocator cap; it does not isolate an allocator-cap speedup or measure the full configuration against original allocator settings at 16K. Its conservative active-peak-plus-allocator-cap bound, including fills/refills, was **12.153 GiB**, below the required **12.32 GiB** ceiling (recommended working set minus 1 GiB). Its snapshot retained about **1,148 MiB**. The long tests cross their first KV-growth boundary; the separate [standalone 16K experiment](../results/inkling_m3/loop-weights-16k-standalone.json) matched all 256 generated tokens across two pairs.

The initial two-pair sampling result had a 0.964 math decode ratio, and the initial five-pair common-cap 2K result had a 0.9669 ratio. Both narrowly failed the 0.97 regression gate. They remain preserved as inconclusive timing evidence; the larger sampling and 2K reruns above passed. The original long analysis therefore still correctly reports its combined gate as false. Acceptance uses the larger 2K rerun and the passing 8K/16K cases, rather than relabeling that earlier report.

## Kernel and memory experiment dispositions

- The full LUT eligibility scan found only layer 1 and layer 6 gate/up sides using staged RTR tables. All tested outputs matched when switching to device tables, but the six-token kernels were about 1.7% and 14.1% slower. No LUT dispatch was changed.
- Recomputing the same FP16 loop weights at 16K instead of retaining all loop-layer pairs preserved all 256 generated tokens across two pairs. Peak active memory decreased from 12.381 to 11.905 GiB, about 3.8%; this missed the standalone 10% memory gate. No default prefill policy was changed.
- Reduced prefill scratch is used only as supporting memory management for the accepted optional larger exact cache. Ordinary prefill policy remains unchanged.
- Dynamic-position single-layer compilation matched the tested outputs and cache arrays but corrected direct/compiled times were 1.794/1.789 ms. The isolated advantage disappeared after fixing reference-wrapper aliasing. Full-model integration is deferred.
- Earlier launch, attention, fixed-width drafting, prompt lookup, allocation, loading and native experiments are recorded individually in the [experiment ledger](../results/inkling_m3/ledger.md). No additional decode optimization has passed the 5% balanced gate.

## Obstructions and limits

- Ordinary sandbox execution cannot access Metal. Model runs and GPU tests used the approved execution path; no model downloads or active dependency changes were needed.
- Xcode trace inspection is unavailable. MLX successfully captured a target call, but the capture duplicated about 9.8 GB of allocated disk space. A decoded [metadata and size manifest](../results/inkling_m3/capture-manifest.json) was saved before removing the generated payload. The manifest cannot replay the capture.
- Capture polluted subsequent synchronized timings. The clean [target profile](../results/inkling_m3/target-profile-clean.json) is the valid diagnostic. Per-component synchronization changes scheduling, and CPU self-times include native/GPU waits; neither supplies normal GPU durations. The available evidence does not justify a full C++ rewrite or full-model compiled-state refactor.
- The 16K case is near the recommended working-set headroom limit. Two uncached warmups failed with Metal out-of-memory callbacks before snapshot creation. Failed checkpoints and logs are retained. A standalone four-run comparison and subsequent bounded-allocator studies succeeded. Startup measurement found 1.72 GB of unused allocator cache after loading, 344 MB after draft-head preparation, and zero after clearing; active bytes were unchanged. These observations do not prove a unique cause of the failures. System swap was already about 1.5–1.6 GiB; thermal reporting supplied no recorded warning, which does not prove constant temperature.
- Exact matching on the frozen suite is finite validation evidence. It does not prove universal numerical equivalence, and existing serial/drafted differences do not by themselves establish a quality loss.

## Validation and reproducibility

The final [release checks](../results/inkling_m3/release-checks.json) passed after exposing both low-memory serving flags: the full Inkling runtime and fast-kernel suite, server suite, and three CLI CUDA tests. Coverage includes snapshot isolation/eviction, greedy and seeded sampling, cache growth, and memory-policy restoration on error and early generator close. Eight model-free benchmark-analysis tests also passed.

Final review found that an error in statistics reporting or evaluation of a pending serial step could bypass memory-limit restoration. A [focused reproducer](../results/inkling_m3/cleanup-regression-before-fix.json) failed before the fix. Both generator cleanup paths now guarantee restoration around reporting/evaluation as well as synchronization. Regression checks cover both generators' reporting errors and pending serial evaluation; the full Inkling/runtime/kernel suite passed again in `release-inkling-cleanup-tests.log`. This changes teardown only; the performance measurements above are retained.

Both real local HTTP retry checks passed: [ordinary exact cache](../results/inkling_m3/exact-cache-live-server.json) and [optional low-memory exact cache](../results/inkling_m3/exact-cache-low-live-server.json). Choices and stopping matched; each retry reported 25 cached prompt tokens. These short-request HTTP checks verify the public configuration path; the long-context performance evidence comes from the generation benchmarks above. All owned benchmark and test-server processes were stopped.

Benchmark entry points are `runtime_mlx.inkling.benchmark`, `runtime_mlx.inkling.bench_exact_prompt_cache`, and `runtime_mlx.inkling.profile_m3`. JSON checkpoints freeze settings, prompt IDs, package/device details and source hashes at measurement time. [Completion metadata](../results/inkling_m3/completion-status.json) records final source hashes, release outcomes and final static/process checks. Analysis can run without importing MLX:

```sh
.venv/bin/python runtime_mlx/inkling/analyze_benchmark.py \
  results/inkling_m3/baseline-balanced-256-complete.json \
  --baseline-arm serial --candidate-arm baseline
.venv/bin/python runtime_mlx/inkling/analyze_exact_cache.py \
  results/inkling_m3/exact-cache-balanced.json
```

For commands, exact cached paths and obstructions, see the [continuation log](inkling-m3-completion-log.md). The [original plan](inkling-m3-optimization-plan.md) and ledger retain all acceptance thresholds and rejected/deferred work. Further decode work would need a consistent measured lead and stronger GPU-duration attribution before a full-model compiled or C++ implementation is justified. Roll back caching with `--prompt-cache 0`; remove `--prompt-cache-low-memory` to use the ordinary allocator/prefill policy.
