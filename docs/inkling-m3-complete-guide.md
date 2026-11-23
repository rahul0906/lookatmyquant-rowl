# Inkling M3 Pro optimization: complete work guide

Written 24 November 2026. This is the entry point for the work in the earlier **Explore C++ model optimization** chat and the continuation in this chat. It combines their objectives, experiments, implementation, final measurements, verification, obstructions and handover information. It was checked against the earlier chat history, the current source, the experiment ledger and the saved results.

The optimization cycle is complete. The accepted change is **exact caching of an identical prompt's original prefill**, with optional memory management for larger snapshots on M3 Pro. **No new decode-throughput change met the required 5% balanced improvement.**

Every source/support file introduced or modified for this work, the related existing runtime files, and every retained file in `results/inkling_m3/` is explained in the [file-by-file map](inkling-m3-file-map.md). That map distinguishes executable runtime code, experimental scripts, tests, documentation, packaging metadata and measurement artifacts. This documentation covers the Inkling optimization work; unrelated compiler, CUDA and other-model implementations are outside its scope.

## 1. Reading order and document responsibilities

| Document | What it explains |
|---|---|
| This guide | The combined story of both chats, why each branch was pursued, what was accepted, how it works and how to use it |
| [File map](inkling-m3-file-map.md) | Each relevant file's purpose, important code and relation to the experiments; complete retained-result inventory |
| [Final results](inkling-m3-results.md) | Performance tables, acceptance evidence, serving options and limits |
| [Experiment ledger](../results/inkling_m3/ledger.md) | Individual historical experiments, their measurements and dispositions |
| [Continuation log](inkling-m3-completion-log.md) | Chronological continuation progress, commands, process sessions, failures and responses |
| [Original plan](inkling-m3-optimization-plan.md) | Constraints, measurement method, acceptance thresholds and final disposition |
| [Completion metadata](../results/inkling_m3/completion-status.json) | Machine-readable final status, source hashes and release/static/process check outcomes |

Historical log entries describe what was known when they were written. An early promising result can later be rejected. The final-results report and final ledger dispositions determine what was accepted. A JSON file with `status: complete` means its workload completed; it does not mean the candidate passed the acceptance rules.

## 2. Original request, environment and rules

The earlier chat began with this command and a question about generating C++ to make the model faster on this Mac:

```sh
lamq generate shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB \
  --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter \
  -p "What is 17 times 23? Show the steps." --stats
```

The request developed into a broader measured optimization plan. C++ export was one candidate alongside attention, kernel launches, drafting, synchronization, prefill, loading and caching.

| Fixed item | Value |
|---|---|
| Machine | Apple M3 Pro, 18-core GPU, 18 GiB unified memory |
| Libraries | MLX 0.32.4; mlx-lm 0.32.2 |
| Target | Inkling-2-Mini-LookAtMyQuant-8.9GB; packed SQYFR text weights approximately 8.87 GB |
| Target revision | `b5fe3bee6a8dc6dd50bb00c2dd17d2e259365a85` |
| Drafter revision | `adac74d1cf00ed68a19a77233a62bd6dcc6c2607` |
| Data access | Cached local model files, `HF_HUB_OFFLINE=1` |
| Prompt setting | `reasoning_effort=none`, matching the original generate command without `--thinking` |
| Output reference for new changes | The existing drafted execution path |

Weights, quantization, tokenizer/template behavior, reasoning settings, FP16 KV and full-target verification were preserved. The work did not retrain, requantize, download a new model, skip target layers/experts, use approximate attention or relax speculative acceptance. An isolated older MLX installation was evaluated separately; the active library version was retained.

Acceptance required identical greedy token IDs and stopping on the applicable frozen suite. Decode changes needed at least 5% family-balanced geometric mean improvement, with no family slowing more than 3%. Startup, prefill, cache or memory changes needed at least 10% improvement in their intended metric, with the same decode-regression limit. Peak addition could not exceed 512 MiB, and peak had to stay at least 1 GiB below MLX's recommended working set. Five interleaved pairs were the normal study size; borderline results were repeated with seven pairs. Profiling was kept separate from performance measurements.

## 3. What the earlier chat did

### Established the reference and diagnosed an existing token difference

The initial benchmark compared serial execution, the manifest-selected drafter and the model-specific 8.9 GB drafter. It found that the existing drafted path and serial path already differed on a coding prompt. A replay used the same serial prefix and cloned cache to compare target logits at several verification widths:

- A single-token forward favored token `149918` over token `13` by 0.05026 before softcap.
- A six-token forward favored token `13` over token `149918` by 0.00252.
- Widths 2, 4 and 8 chose the serial token in that replay.

This identified width-dependent target arithmetic as sufficient to explain that example. Full target verification does not guarantee the same greedy sequence as serial when different floating-point execution shapes change a close decision. The cache optimization therefore compares with the existing drafted reference, and does not claim to repair every serial/drafted difference or establish which answer is more accurate.

The earlier chat created the plan, common benchmark, paired analyzer, specialized microbenchmarks, diagnostic scripts and ledger. It ran the following branches. Ratios are candidate/reference; small exploratory or isolated results do not establish production speedups.

| Experiment | What happened | Final disposition / evidence |
|---|---|---|
| Specialized 8.9 GB drafter | Balanced ratio 1.006; changed output on three of nine prompts | Rejected; [drafter-balanced-analysis.json](../results/inkling_m3/drafter-balanced-analysis.json) |
| Forced fused attention | Changed code/chat outputs and was slower in exploratory runs | Rejected; [attention-force-smoke.json](../results/inkling_m3/attention-force-smoke.json) |
| Single-token MoE launch geometry | Sampled kernels matched; best isolated gains 0.9–4.5%; initial run drifted before interleaving was corrected | Inconclusive for generation; [moe-launch-interleaved.json](../results/inkling_m3/moe-launch-interleaved.json) |
| Fixed draft lengths 2/4/6/8 | Some workloads sped up, but every fixed choice changed an applicable output | Rejected; [draft-lengths.json](../results/inkling_m3/draft-lengths.json) |
| Grouped MoE rows per group = 2 | Matched outputs; balanced end-to-end gain 0.78% | Below gate; [moe-rps2-analysis.json](../results/inkling_m3/moe-rps2-analysis.json) |
| Disable target async submissions | Math slowed about 6%; stopped early | Rejected; [core-async-off-smoke.json](../results/inkling_m3/core-async-off-smoke.json) is partial |
| Drafter block trimming | Matched outputs; ratio 0.9955 | Rejected; [drafter-trim-analysis.json](../results/inkling_m3/drafter-trim-analysis.json) |
| Cold long-context reference | One short-output pass per mode at approximately 2K/8K/16K; 16K near memory ceiling | Exploratory reference; [long-cold.json](../results/inkling_m3/long-cold.json) |
| 512-token prefill chunks at 8K | Saved memory, increased first-token time and changed token 8 | Rejected; [prefill-512-8k.json](../results/inkling_m3/prefill-512-8k.json) |
| Six-row vocabulary-head launch | Sampled head call improved; end-to-end gain only 1.05% | Below gate; [head-launch-analysis.json](../results/inkling_m3/head-launch-analysis.json) |
| Existing prefix prompt cache | Approximately 5.01 s to 0.063 s repeated prefill at 1,974 tokens, but changed output | Rejected under strict parity; [prompt-cache.json](../results/inkling_m3/prompt-cache.json) |
| Async submission every eight visits | Matched output; ratio 1.001 | Below gate; [eval-stride8-analysis.json](../results/inkling_m3/eval-stride8-analysis.json) |
| GPU greedy-prefix matcher | Boundary checks/output parity passed; ratio 0.992 | Removed from runtime; [gpu-greedy-verify-analysis.json](../results/inkling_m3/gpu-greedy-verify-analysis.json) |
| Async submission every two visits | Matched output; ratio 1.003 | Below gate; [eval-stride2-analysis.json](../results/inkling_m3/eval-stride2-analysis.json) |
| Compiled drafter recurrence | Matched output; ratio 1.012 | Benchmark-only; [compiled-draft-analysis.json](../results/inkling_m3/compiled-draft-analysis.json) |
| Read-only NPZ mapping | Matched sample output; loading 16.42 s versus 6.33 s | Rejected early; [load-mapped-1.json](../results/inkling_m3/load-mapped-1.json) |
| Direct reads of stored NPZ members | Loading 6.37 s versus 6.33 s; prototype skips ZIP CRC checks | No startup gain; benchmark-only; [load-direct-1.json](../results/inkling_m3/load-direct-1.json) |
| Thresholded fused attention at 8K | Matched output; 16.260 versus 16.273 tok/s in one exploratory case | Below gate; [fused-threshold-8k.json](../results/inkling_m3/fused-threshold-8k.json) |
| Six-row QKV launch | Matched sample/end-to-end outputs; balanced gain about 0.82% | Below gate; [qkv-launch-analysis.json](../results/inkling_m3/qkv-launch-analysis.json) |
| Held-out serial/drafted reference | All three held-out drafted sequences differed from serial | Existing limitation recorded; [heldout-baseline.json](../results/inkling_m3/heldout-baseline.json) |
| Geometric FP16 KV allocation | Initial cold result looked faster; five warmed pairs both measured 17.35 tok/s | Rejected; [kv-growth-8k-paired.json](../results/inkling_m3/kv-growth-8k-paired.json) |
| Native C++ drafter recurrence | Export/import matched; Python 2.290 ms, C++ 2.300 ms in separate timing runs | Feasibility without a gain; [native-draft-python.json](../results/inkling_m3/native-draft-python.json), [C++](../results/inkling_m3/native-draft-cpp.json) |
| Isolated MLX 0.32.3 | Matched smoke outputs; non-interleaved timing gave no improvement | Active 0.32.4 retained; [mlx-0323-smoke.json](../results/inkling_m3/mlx-0323-smoke.json) |
| Native empty-cache target layer | Export initially wrong because of non-contiguous constants; materializing views restored exact sample outputs | Fixed-shape feasibility only; [native-layer-python.json](../results/inkling_m3/native-layer-python.json), [C++](../results/inkling_m3/native-layer-cpp.json) |
| Native nonempty-cache target layer | Explicit KV/convolution inputs matched for two input sets; timings varied | Position-64/layer-0 feasibility only; [native-stateful-python.json](../results/inkling_m3/native-stateful-python.json), [C++](../results/inkling_m3/native-stateful-cpp.json) |
| Pre-materialized attention weights | Matched generated output; ratio 1.0032 | No runtime promotion; [contiguous-weights-analysis.json](../results/inkling_m3/contiguous-weights-analysis.json) |
| Verified prompt lookup | Engaged on copying cases; changed code output; ratio 0.991 | Rejected; [ngram-copy-analysis.json](../results/inkling_m3/ngram-copy-analysis.json) |
| Attention-output projection launches | Some layers improved; layer 0 slowed about 5% | Inconclusive for generation; [dense-o-six.json](../results/inkling_m3/dense-o-six.json) |
| In-process compiled stateful layer | Two inputs matched; isolated 1.543 to 1.475 ms; Python positions retraced | Fixed-position feasibility; [stateful-compiled-layer.json](../results/inkling_m3/stateful-compiled-layer.json) |
| Dynamic-position compiled sliding layer | One trace at positions 0/1/32/64/96/250; corrected independent-reference timings 1.794/1.789 ms | Effectively flat; [dynamic-layer-boundaries.json](../results/inkling_m3/dynamic-layer-boundaries.json) |

The earlier chat also corrected a drafter-path reproducibility problem: resolving a Hugging Face symlink all the way to an extensionless blob produced a path MLX could not reopen directly. Result files now preserve a usable cached filename. It strengthened the analyzer to reject incomplete or duplicate comparisons, then began a longer balanced reference. Usage exhaustion interrupted that run after 35 of 90 planned rows. The partial file is preserved and was never combined with later measurements.

## 4. What this continuation did

1. **Recovered the work and completed the baseline.** Reviewed the documentation and earlier experiments, kept their files, and ran a fresh nine-prompt/five-repeat/256-token serial/drafted baseline. It contains 90 measured runs, with stable output within each mode. The harness records prompt IDs, settings, device/library details and source hashes, writes atomic checkpoints and marks incomplete normal interpreter exits as interrupted. The analyzer gained explicit reference-arm selection and rejection of non-finite or non-positive throughput.
2. **Built the exact retry cache.** The earlier prefix cache splits the prefill at `n-1`, which changed real-model outputs in its test. The new cache saves the original complete prefill and its first-token result. It preserves target state, final taps and the matching drafter handoff. Miniature regression checks, a 30-run smoke study and a 90-run balanced study passed.
3. **Profiled separately and tested remaining leads.** Captured MLX Metal work, preserved a capture manifest, removed the large owned capture payload and repeated profiling without capture. Scanned every gate/up and down MoE side for LUT staging; only two of 48 were eligible. Their changed setting was slower. No launch dispatch was changed.
4. **Investigated 16K memory failures.** Two uncached warmups failed before a snapshot existed. A standalone loop-weight comparison succeeded and saved about 3.8% active peak memory, below the standalone 10% gate. Startup measurements identified unused allocator cache. These observations informed an optional M3 Pro policy that recomputes the same FP16 loop weights and temporarily bounds unused allocator cache.
5. **Completed broader validation.** Ran held-out prompts, original/extended/edited histories, seeded sampling and five pairs at 2K/8K/16K. The initial sampling math ratio and 2K ratio narrowly failed the 3% regression rule. Seven-pair reruns passed. An additional five-pair 8K study compared the whole optional configuration against original allocator settings. Earlier failing analyses remain preserved.
6. **Exposed and checked serving.** Exported `ExactPromptCache` for Python and added explicit exact-mode, retained-budget and low-memory flags to the server. Tested real local HTTP retries through both ordinary and low-memory public options. Each retry returned identical choices/stopping and reported 25 cached prompt tokens.
7. **Fixed a final cleanup edge case.** An exception during final statistics reporting or evaluation of a pending serial step could skip memory-limit restoration. A focused test reproduced the failure. Both generation cleanup paths now guarantee restoration around those operations. Regression checks cover serial/drafted reporting, pending serial evaluation, synchronization failure and early close. The full runtime/kernel suite passed again.
8. **Finished the handover.** Updated the results, plan disposition, ledger, chronological log, README and completion metadata. Verified syntax, whitespace and local documentation links, and confirmed the owned benchmarks/test servers had exited. This follow-up adds the combined guide and complete file map; it does not change inference code or rerun performance measurements.

## 5. How the accepted implementation works

### Runtime responsibilities

| File | Role in the accepted path |
|---|---|
| [server.py](../runtime_mlx/server.py) | Parses cache options, creates the selected cache, keeps generation on the model's persistent worker and reports reused prompt tokens |
| [generate.py](../runtime_mlx/inkling/generate.py) | Runs prefill and serial/drafted streaming; applies/restores scoped allocator limits; pushes final taps into the drafter |
| [exact_prompt_cache.py](../runtime_mlx/inkling/exact_prompt_cache.py) | Matches exact prompt/settings keys, stores bounded snapshots and restores independent cache wrappers/context |
| [model.py](../runtime_mlx/inkling/model.py) | Executes the original target and maintains KV/convolution state; the new default-true flag controls retaining dequantized loop weights during prefill |
| [drafter.py](../runtime_mlx/inkling/drafter.py) | Existing drafter context snapshot/restore and proposal generation, reused without a production code change |
| [__init__.py](../runtime_mlx/inkling/__init__.py) | Exports the new cache class for Python callers |

On a **miss**, the cache resets the drafter as usual and calls the original full-prefill helper. It snapshots:

- The first-token output: greedy output or ordered logits, depending on execution mode.
- Every target cache array and the position/length state needed to continue.
- The final hidden taps and their starting position.
- The drafter context before the caller consumes those final taps.

On a **hit**, it restores a clone of target state, restores drafter context and returns independent MLX wrappers for output/taps. Generation then consumes the final taps and continues through the original target-verification path. Sampling happens after the cached prefill, so a hit does not save or reuse a random draw.

Matching includes prompt token IDs, ordered/greedy mode, the requested chunk setting, tap visits, relevant execution-environment settings and effective low-memory mode. Cache ownership also checks model/drafter identity. Extensions and edited histories do a fresh prefill. The cache is confined to the model's execution thread; clear it if model state is replaced in place.

Retention defaults to one entry and 512 MiB of retained arrays. Least-recently-used entries are evicted to satisfy count and byte limits. A snapshot exceeding the byte budget is not stored. Retained-array size and measured peak active memory are separate quantities: a byte budget is not a promise about total process memory.

Optional `low_memory=True` is effective only with fast execution on **Apple M3 Pro**. It recomputes identical FP16 prefill loop matrices instead of retaining every loop-layer pair. During generation it caps unused allocator cache at 256 MiB, honors an already tighter limit and clears unused buffers. Nested cleanup restores previous allocator and wired limits when generation completes, is cancelled or raises, including failures during cleanup. Ordinary prefill/allocator defaults remain as before.

## 6. Final results and what they mean

### Existing decode baseline

| Family | Serial median | Existing drafted median |
|---|---:|---:|
| Math | 28.05 tok/s | 41.72 tok/s |
| Coding | 27.59 tok/s | 38.16 tok/s |
| Chat | 26.70 tok/s | 26.21 tok/s |

These describe existing execution modes. They are not a newly achieved speedup. Source: [complete 256-token baseline](../results/inkling_m3/baseline-balanced-256-complete.json).

### Accepted ordinary exact cache

| Family | Typical uncached first token | Cached first token | Decode ratio |
|---|---:|---:|---:|
| Math | 264.87 ms | 3.09 ms | 0.9991 |
| Coding | 230.59 ms | 3.32 ms | 1.0009 |
| Chat | 220.18 ms | 3.16 ms | 0.9937 |

All 90 measured runs matched tokens/stopping and passed the cache gate. Geometric family first-token improvements were about 81x/73x/72x. Whole-response gains on the up-to-256-token short cases were about 0.8–7%. Decode throughput was effectively unchanged. Maximum measured active-peak addition, including fills, was 384 bytes. Source: [balanced cache analysis](../results/inkling_m3/exact-cache-balanced-analysis.json).

### Optional larger low-memory exact cache

| Context | Uncached first token | Cached first token | Decode ratio | Conditions |
|---|---:|---:|---:|---|
| 2K | 4.923 s | 27.454 ms | 1.0025 | Seven pairs against original allocator settings |
| 8K | 20.853 s | 42.769 ms | 1.0061 | Five pairs against original allocator settings |
| 16K | 43.134 s | 36.336 ms | 0.9944 | Five pairs with a 256 MiB allocator cap in both arms |

The 16K study's conservative active-peak-plus-allocator-cap bound was **12.153 GiB**, including fills/refills, below the required **12.32 GiB** ceiling. Its snapshot retained about **1,148 MiB**. The 128-token long studies cross their first KV-growth boundary; a separate standalone 16K experiment matched all 256 tokens over two pairs.

The 16K timing validates caching under a common allocator cap. It does not isolate an allocator-cap speedup or compare the entire configuration with original allocator settings at 16K. The initial combined long analysis still reports a failed gate because of its borderline 2K slowdown. Acceptance uses the larger passing 2K rerun and passing 8K/16K cases; no earlier report was silently relabeled.

Held-out validation matched all 30 runs. History validation matched all 18 runs. Seven-pair seeded sampling at temperature 0.7 matched all 42 runs, with family decode ratios 1.0014/0.9896/1.0051. See [final results](inkling-m3-results.md) for their individual source links.

## 7. How to use it, and which workloads benefit

| Your workflow | Effect of the accepted change |
|---|---|
| A new `lamq generate` process for every prompt | The snapshot does not persist between processes; no new decode gain was established for this workflow |
| An identical retry in a persistent exact-cache server/Python process | Reuses resident full-prefill state and reaches the first token much sooner |
| A conversation with added messages or an edited prompt | Exact mode runs fresh prefill; the original target/drafter then generate normally |
| First request, evicted snapshot or prompt exceeding the budget | Runs fresh prefill; no exact-cache latency saving |
| A long completion after a cache hit | Prefill latency improves; decode throughput remains close to the original path |

Run from the repository root using the frozen local paths:

```sh
INKLING_MODEL_PATH='/Users/rahul/.cache/huggingface/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-8.9GB/snapshots/b5fe3bee6a8dc6dd50bb00c2dd17d2e259365a85'
INKLING_DRAFTER_PATH='/Users/rahul/.cache/huggingface/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-drafter/snapshots/adac74d1cf00ed68a19a77233a62bd6dcc6c2607'

HF_HUB_OFFLINE=1 .venv/bin/lamq serve "$INKLING_MODEL_PATH" \
  --drafter "$INKLING_DRAFTER_PATH" --reasoning-effort none \
  --prompt-cache 1 --prompt-cache-mode exact
```

For the tested larger snapshots, add `--prompt-cache-mib 1536 --prompt-cache-low-memory`. Server cache reuse retains the existing trusted single-user loopback restriction: requiring a bearer key or binding to a network address disables reuse. These flags are MLX options. The default server still has caching disabled and selects the legacy prefix mode if caching is requested without a mode.

For Python, create `ExactPromptCache()` from `runtime_mlx.inkling` and pass it as `prompt_cache` to `stream_dspark` or `stream_serial`. Keep that instance alive on the same model execution thread across requests. For the optional larger configuration use `ExactPromptCache(max_entries=1, max_bytes=1536 * 1024**2, low_memory=True)`. The general `Runtime.stream` API was not extended with a new cache option in this work.

Disable snapshots with `--prompt-cache 0`. Remove `--prompt-cache-low-memory` for the ordinary memory policy. No new kernel setting or compiled/native runner needs to be enabled.

## 8. Tests, reproducibility and artifact interpretation

Final checks passed for the full Inkling runtime/fast-kernel suite, server suite, three CLI CUDA tests, eight model-free analysis tests and real ordinary/low-memory HTTP retries. The final teardown fix was followed by another full runtime/kernel run. [release-checks.json](../results/inkling_m3/release-checks.json) retains the original release outcomes and the additional cleanup-test entry. [completion-status.json](../results/inkling_m3/completion-status.json) records 15 checked Python sources, documentation checks and process cleanup evidence.

The miniature cache tests cover greedy/sample behavior, repeated/extended/edited prompts, chunk keys, execution-setting invalidation, wrapper isolation, entry eviction, byte budgets and disabled caching. Failure-injection tests check memory-limit restoration. Real-model benchmarks supply the broader parity/performance evidence; miniature tests alone do not prove it.

Benchmark files contain frozen settings, prompt IDs, versions, device data, token outputs, stopping, timing and memory fields. Older exploratory artifacts may lack fields added later. Source hashes in benchmark checkpoints refer to measurement-time code; completion metadata records final code after the teardown fix. The file map explains which file is raw data, derived analysis, log or profile.

The general analyzer computes the median of paired candidate/reference throughput ratios per prompt, geometric means within families, then the geometric mean across families. It rejects missing/duplicate repetitions, incomplete suites and invalid throughput. The exact-cache analyzer additionally requires a real hit, first-token gains, matching/stable outputs, memory limits including fills/refills and acceptable family decode ratios.

To analyze without Metal, execute the analyzer by file path:

```sh
.venv/bin/python runtime_mlx/inkling/analyze_benchmark.py \
  results/inkling_m3/baseline-balanced-256-complete.json \
  --baseline-arm serial --candidate-arm baseline
.venv/bin/python runtime_mlx/inkling/analyze_exact_cache.py \
  results/inkling_m3/exact-cache-balanced.json
```

To reproduce the ordinary cache study, use the path variables above:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m runtime_mlx.inkling.bench_exact_prompt_cache \
  "$INKLING_MODEL_PATH" "$INKLING_DRAFTER_PATH" \
  --suite balanced --repetitions 5 --max-tokens 256 \
  --out results/inkling_m3/local-cache-rerun.json
```

Use a new output name to preserve earlier evidence. GPU runs need Metal access and should run one at a time. Exact long-validation commands are retained in [final-validation-progress.json](../results/inkling_m3/final-validation-progress.json) and [additional-validation-progress.json](../results/inkling_m3/additional-validation-progress.json). Some older prototype scripts assume fixed shapes, local result files or temporary export paths; consult their file-map entry before running them.

## 9. Obstructions, responses and remaining limits

| Obstruction or uncertainty | Response and evidence |
|---|---|
| Prior chat exhausted usage during the long baseline | Preserved the incomplete 35-run checkpoint; completed a fresh 90-run reference |
| Restricted shell lacked Metal access | Used approved execution for GPU work; read-only JSON analysis ran by file path |
| `xctrace` unavailable | MLX capture succeeded; full GPU-duration inspection remains unavailable |
| Capture duplicated about 9.8 GB allocated disk space, 18.63 GB logical, 2,930 files | Saved decoded metadata/size [manifest](../results/inkling_m3/capture-manifest.json), then removed the owned trace payload; manifest cannot replay it |
| Capture polluted synchronized component timings | Repeated without capture; use [target-profile-clean.json](../results/inkling_m3/target-profile-clean.json) |
| CPU self-time and synchronized component durations are not normal GPU durations | Documented their scope; did not use them to justify a full C++ rewrite |
| Two 16K uncached warmups hit Metal out-of-memory before snapshot creation | Saved failed checkpoints/logs; standalone and bounded studies subsequently succeeded; unique failure cause is unproven |
| Unused allocator memory after startup | [startup-memory.json](../results/inkling_m3/startup-memory.json): 1.72 GB after loading, 344 MB after draft-head preparation, zero after clearing; active bytes unchanged |
| System swap already about 1.5–1.6 GiB; no recorded thermal warning | Retained as system context; neither proves constant temperature or that the model caused swapping |
| Initial small sampling and 2K studies narrowly slowed more than 3% | Preserved failures; larger reruns passed; reference conditions labeled explicitly |
| Final cleanup could leave temporary memory limits applied after an error | Preserved the failing reproducer, fixed both cleanup paths and reran the full suite |
| Process inventory blocked in the ordinary sandbox | Approved read-only inspection found no owned jobs or listeners; later test session also exited |
| Documentation follow-up's full staged whitespace check reported spaces in generated `lookatmyquant.egg-info/PKG-INFO` | New/updated documentation passed its own whitespace/link checks; existing generated metadata and staging were preserved |

Finite parity tests do not prove universal numerical equivalence or model quality. A new comprehensive quality evaluation was not performed. The earlier cold 16K serial/drafted comparison was exploratory; no stronger new 16K serial-versus-drafted quality study was completed. Long-context cache comparisons use the existing drafted reference.

Full-model C++/compiled integration and deeper GPU-duration attribution remain deferred. They would need usable trace inspection, a consistent measured performance lead, dynamic cache/metadata support and complete generation parity checks. The corrected dynamic-layer timing currently supplies no such lead. The accepted cache improves a specific real workload; this work does not establish that every possible optimization has been exhausted.

## 10. Current workspace and maintenance handover

The runtime and experimental files remain in the existing checkout. No optimization commit, push, new worktree or automation was created by these chats. Staging state may reflect subsequent user actions and should be preserved. The earlier environment setup generated `.venv/` and `lookatmyquant.egg-info/`; these are installation artifacts, explained in the file map.

No owned benchmark or test server was left running at completion. No heavyweight Metal trace payload is retained; its manifest is retained. Temporary native exports and the isolated MLX environment are historical experiment paths, not production dependencies, and their continued presence is not assumed by this guide.

For future changes, retain the recorded baseline, compare with the same execution mode, preserve failing evidence, include fill/refill memory and repeat borderline timings before promotion. Update the ledger and final-results report when a new measured disposition changes. Update the file map if source or result files are added. The existing completion metadata remains an optimization-completion snapshot; this documentation follow-up does not rewrite measurement-time source hashes.
