# Inkling M3 experiment ledger

## Cached specialized drafter: inconclusive

Two exploratory paired repetitions, 128 token cap, one prompt each for math, code and chat; both drafters resident in memory. Results: [drafter-smoke.json](drafter-smoke.json). The manifest-selected drafter and the 8.9 GB-specific drafter achieved respectively 45.68/45.54, 29.62/31.60, and 23.91/24.49 median tokens/s. This is too small a sample for the acceptance criteria. The reference drafted path diverged from serial at generated token 85 on the coding prompt in both runs; the specialized candidate matched serial on these three prompts. Diagnose the target verification/rounding before promoting changes. Serial-versus-drafted output differences may arise from differing numerical paths and do not by themselves identify the cause.

Both drafters are loaded simultaneously, so peak memory is not a fair per-drafter comparison. First-use measurements include earlier warmup and are not cold-start measurements.

## Experiment status

The optimization cycle completed on 23 November 2026. Exact repeated-prompt caching is accepted and available through explicit serving/Python options, including a validated optional M3 Pro memory policy for larger snapshots. Balanced, held-out, history, sampled, long-cache and public HTTP checks completed; final runtime/kernel, server and CLI suites passed. No new decode-throughput change met the 5% balanced gate. Full-model compiled/C++ integration and Xcode GPU-duration inspection remain deferred. Historical entries below preserve their original measurements and limitations; final continuation dispositions follow at the end. See the [final report](../../docs/inkling-m3-results.md).

## Divergence replay

Replayed the serial prefix for the coding prompt and compared logits at the first divergent token using identical cloned cache state. In a single-token forward, token 149918 scored 40.54981 and token 13 scored 40.49955 (gap 0.05026, before softcap). In a width-6 forward, token 13 scored 40.56245 and token 149918 40.55993 (gap 0.00252). Widths 2, 4, and 8 chose the serial token in this replay. This isolates width-dependent target computation as a sufficient cause for this particular divergence; it does not rule out additional causes elsewhere. Replay using `python -m runtime_mlx.inkling.diagnose_divergence results/inkling_m3/drafter-smoke.json` on the cached model. Do not describe speculative execution as universally token-identical to serial without qualifying numerical differences.

## Forced fused attention: rejected for current acceptance gate

Two exploratory repetitions per prompt, separately from the prior run; results in [attention-force-smoke.json](attention-force-smoke.json). Compared to the earlier default-attention smoke run, forced attention changed serial tokens on code (first difference at index 84) and chat (index 63), and changed drafted tokens on those prompts as well. It also reduced measured drafted throughput on all three prompts (45.68 to 41.78, 29.62 to 27.14, and 23.91 to 23.16 tokens/s). These are non-interleaved exploratory results, not a definitive performance estimate, but output differences alone fail the agreed gate. Do not enable the forced option.

## M3 single-token MoE launch sweep: inconclusive

The [interleaved microbenchmark](moe-launch-interleaved.json) runs gate/up and down kernels with actual packed weights from layers 0, 12, and 23, using one deterministic input and eight expert IDs. It warms each launch geometry and alternates the measurement order over 21 rounds. All ten tested geometries produced bit-identical output to the current launch setting for these inputs. The fastest median in each case was between 0.9% and 4.5% faster than the current setting. These are isolated, synchronized kernel-call timings including host overhead, not end-to-end generation; the small gains, limited input coverage, and observed startup drift do not establish a deployable improvement. No dispatch change has been enabled. The earlier [non-interleaved run](moe-launch-microbench.json) used a less representative gate/up input layout and exhibited timing drift; it is retained as diagnostic data and must not be used for tuning decisions.

Reproduce with `python -m runtime_mlx.inkling.bench_m3_moe <local-model-directory> --rounds 21 --out results/inkling_m3/moe-launch-interleaved.json`. Test additional expert selections, all layers, verification widths and end-to-end token parity before considering a launch profile.

## Cached specialized drafter: rejected after balanced comparison

Five repetitions over nine math, code, and chat prompts, 128-token cap; [raw runs](drafter-balanced.json) and [paired analysis](drafter-balanced-analysis.json). The specialized/default family-balanced speed ratio was 1.006 (math 0.983, code 0.996, chat 1.039), below the 1.05 gate. Candidate tokens differed from the existing drafted path on three of nine prompts, independently failing the exact-output gate. The baseline outputs were stable across repetitions. Both drafters were resident, so this is not a standalone memory comparison. Keep the manifest-selected default.

## Fixed draft lengths: rejected

Five interleaved repetitions with the default drafter on one prompt per family; [results](draft-lengths.json). Fixed width 6 improved math median 45.07 to 47.72 tok/s with identical math output, but changed both coding and chat outputs. Widths 2 and 4 improved coding and chat by approximately 15–30%, but also changed their outputs; width 8 was slower and changed code. Every fixed length therefore fails the output-parity gate. Further tuning should retain the current block choices and focus on accelerating the corresponding target computations.

## Grouped MoE rows-per-group=2: rejected

The [six-token microbenchmark](moe-grouped-six-rps.json) found bit-identical outputs for sampled inputs and improvements on some layers. Five paired end-to-end runs per smoke prompt showed [only 0.78% family-balanced gain](moe-rps2-analysis.json), below the 5% acceptance gate; all three generated sequences matched their baseline. Peak MLX memory was 9.57 GB for both arms. No runtime dispatch change was made.

## Disable target-forward async submissions: rejected early

Two paired math runs showed 45.27/45.13 tok/s baseline versus 42.31/42.47 tok/s candidate (approximately 6% slower). This exceeds the permitted family regression, so the smoke run was interrupted after the completed math case, per the plan's early-stop rule. [Partial data](core-async-off-smoke.json) are marked running because the process was interrupted; no runtime change was made.

## Drafter block trimming: rejected after smoke test

Two paired runs for one prompt per family. [Results](drafter-trim-smoke.json), [analysis](drafter-trim-analysis.json). Tokens matched the existing drafted output on all three cases, but family-balanced throughput was 0.9955x baseline, so the candidate fails the performance gate and was not promoted to a full suite.

## Long-context baseline: exploratory

One cold/unwarmed pass per arm, with 16 generated tokens, on synthetic repeated research notes. At 1,974 / 8,118 / 16,310 prompt tokens, the drafted path reached first-token times 4.83 / 20.83 / 42.36 seconds and peak MLX memory 10.82 / 11.91 / 12.38 GiB. Serial was run first in each case, including cold compilation for the first case, so do not compare its initial latency directly. At 16K, the drafted peak is slightly above the plan's limit of 1 GiB below the 13.32 GiB recommended working set; treat it as a stress case. The drafted output differed from serial at 16K. [Raw data](long-cold.json).

## 512-token prefill at 8K: rejected

Two paired exploratory repetitions, 16 generated tokens, unwarmed long prompt. [Raw runs](prefill-512-8k.json). The existing auto path reached first token in 21.22 / 21.15 seconds with 11.84 / 11.91 GiB peak MLX memory; 512-token chunks took 23.59 / 23.45 seconds and 11.01 GiB. Candidate output changed from the existing drafted path at generated token index 7 in both runs, independently failing the exact-output gate. No setting was changed.

## Six-row vocabulary-head launch: rejected

On one sampled input, the [isolated head kernel](head-six.json) improved from 3.69 ms to 3.11 ms with identical output. Five paired runs per smoke prompt produced only a 1.05% family-balanced end-to-end gain, with exact output parity; [analysis](head-launch-analysis.json) and [raw results](head-launch-smoke.json). This fails the 5% gain gate, so no runtime dispatch was changed. The isolated test did not cover all possible inputs.

## Prompt cache: rejected for output parity

One exploratory run on a 1,974-token synthetic prompt: [raw results](prompt-cache.json). The warmed uncached prefill took 5.01 seconds; a cache hit reused 1,973 tokens and took 0.063 seconds in prefill. The cache fill and hit both diverged from uncached greedy output at generated token index 25; a second uncached run matched the original. Its split prefill changes the numerical execution shape. The cache is already available through `lamq serve --prompt-cache`, but it is not recommended under this task's strict output-parity requirement. No default was changed. [Reproduction](../../runtime_mlx/inkling/bench_prompt_cache.py).

## Eight-visit async submissions: rejected

Two exploratory paired runs per smoke prompt showed a 1.001 family-balanced speed ratio with identical token sequences: [results](eval-stride8-smoke.json), [analysis](eval-stride8-analysis.json). This is below the 1.05 threshold and was not enabled.

## GPU greedy-prefix matcher: rejected

An experimental compiled MLX prefix matcher passed tests for zero, partial, and full acceptance and matched existing drafted tokens on three smoke prompts. Two paired runs per prompt showed a 0.992 family-balanced speed ratio, so the candidate was removed from the runtime. [Raw results](gpu-greedy-verify-smoke.json) and [analysis](gpu-greedy-verify-analysis.json).

## Two-visit async submissions: rejected

Two paired runs per smoke prompt produced a 1.003 family-balanced speed ratio and matched outputs. This is below the 1.05 threshold. [Raw results](eval-stride2-smoke.json), [analysis](eval-stride2-analysis.json).

## Compiled drafter recurrence: rejected at smoke stage

The experimental fixed-length compiled MLX graph matched existing drafted output across three prompts, but two paired repetitions showed a 1.012 family-balanced speed ratio, below the 1.05 gate. [Results](compiled-draft-smoke.json), [analysis](compiled-draft-analysis.json). It remains a benchmark-only experiment.

## Read-only NPZ memory mapping: rejected early

A small synthetic NPZ parity check passed, and one full-model mapped run matched the standard run's generated tokens. However, loading took 16.42 seconds versus 6.33 seconds with the standard loader, so the next run was interrupted under the regression rule. [Baseline](load-baseline-1.json), [mapped](load-mapped-1.json). No model files were modified and no runtime path was enabled.

## Direct reads of uncompressed NPZ members: rejected at smoke stage

One exploratory run loaded in 6.37 seconds, versus 6.33 seconds for the prior normal-loader run; generated tokens matched. [Direct result](load-direct-1.json), [baseline](load-baseline-1.json). This is not a paired multi-run study, but it shows no evidence near the 10% startup gate. The experiment is benchmark-only and skips ZIP CRC verification, so it must not be promoted to a production loader without preserving equivalent integrity checks.

## Thresholded fused attention at 8K: rejected at smoke stage

One exploratory 8,118-token prompt with 128 generated tokens: [results](fused-threshold-8k.json). The baseline and thresholded fused setting matched tokens, with 16.260 and 16.273 tok/s respectively. This is not a 5% gain and is too small for the full acceptance study. The option remains off by default on this M3.

## Six-row QKV launch: rejected at smoke stage

The [layer 0](qkv-six-0.json), [layer 12](qkv-six-12.json), and [layer 23](qkv-six-23.json) kernel sweeps found a `(rb=2, sg=4, threadgroups=160)` configuration that matched sampled kernel outputs and produced small isolated gains. Two paired end-to-end repetitions per smoke prompt retained generated tokens but delivered only a small overall speed difference; [results](qkv-launch-smoke.json), [analysis](qkv-launch-analysis.json). It did not meet the 5% threshold, and runtime dispatch is unchanged.

## Held-out serial versus drafted reference

One run each on three held-out math, coding, and chat prompts: [data](heldout-baseline.json). All three drafted sequences diverged from their serial counterparts. These are pre-existing baseline differences and do not establish whether either answer is less accurate. Candidate optimization comparisons use the existing drafted path as the fixed reference.

## Geometric FP16 cache growth at 8K: rejected

The initial un-warmed candidate appeared 7.5% faster in decode, but five warmed, alternating pairs showed a median 17.35 tok/s for both existing and geometric growth, with matching generated outputs. [Paired results](kv-growth-8k-paired.json). First-token latency and memory did not meet the 10% improvement gate. The runtime retains the existing allocation policy.

## Native C++ drafter recurrence: no isolated gain

Exported the real six-token greedy recurrence and 32K quantized shortlist head as a 74.1 MB MLX function, using deterministic synthetic hidden/context input. C++ imported the graph and exactly matched the Python output. Twenty synchronized warm samples had medians 2.290 ms in [Python](native-draft-python.json) and 2.300 ms in [C++](native-draft-cpp.json), measured sequentially rather than interleaved. This tests only the recurrent proposal subgraph, not target execution, full generation, or Python graph construction. It does not establish that a full C++ runner would be equally fast; this subgraph alone gives no evidence for a rewrite. Sources: [export](../../runtime_mlx/inkling/bench_native_export.py), [C++ runner](../../runtime_mlx/inkling/bench_native_draft.cpp).

## MLX 0.32.3 isolated comparison: inconclusive/rejected for promotion

Installed mlx and mlx-metal 0.32.3 in `/private/tmp/inkling-mlx-0323` without changing the active 0.32.4 environment. Two exploratory smoke runs per prompt matched the earlier 0.32.4 serial and drafted output on all three prompts. The drafted rates were approximately 6% lower than the earlier 0.32.4 smoke run, but the runs were not interleaved and machine temperature/workload may differ; this is not evidence that the library itself is slower. There is no evidence of the required improvement, so it is not promoted. [Results](mlx-0323-smoke.json).

## Native target-layer feasibility

Exported one real six-token target layer with all five resulting cache arrays. The first attempt changed outputs substantially: tracing isolated the first difference to non-contiguous attention convolution weights, and later a transposed relative-position weight. Materializing these views with `mx.contiguous` before export restored bit-identical intermediate results, final layer output, and cache arrays for this input. C++ imported the graph and matched all six outputs. [Python direct versus exported timings](native-layer-python.json), [C++ timing](native-layer-cpp.json). In 20 alternating Python calls, median direct layer time was 1.420 ms versus 1.365 ms for imported replay (3.8%); C++ replay was 1.559 ms in a separate run and cannot be compared reliably due to non-interleaving. This is a single layer with empty cache and synthetic input, using a 281.6 MB fixed-shape graph. It does not yet support arbitrary cache contents, positions, or the full model and is not an accepted runtime optimization.

The export reproducibility scripts are [layer Python](../../runtime_mlx/inkling/bench_native_layer.py), [layer C++](../../runtime_mlx/inkling/bench_native_layer.cpp), and [constant-view diagnosis](../../runtime_mlx/inkling/diagnose_export_constants.py). The exported graph stays in `/private/tmp` rather than the repository.

## Native target layer with nonempty cache

Exported the six-token layer-0 forward with six explicit inputs: hidden activations, a populated FP16 KV buffer, and four convolution-state arrays at fixed position 64. For two different sets of inputs, all six output arrays matched direct Python execution bit for bit. A C++ runner also matched the saved outputs. Across two exploratory alternating 20-sample Python runs, direct/replay medians were 1.899/1.730 ms and 1.630/1.556 ms; timings were noisy. The C++ run had 1.491 ms median but was separate, not interleaved. [Python results](native-stateful-python.json), [C++ results](native-stateful-cpp.json), [Python prototype](../../runtime_mlx/inkling/bench_native_stateful.py), [C++ prototype](../../runtime_mlx/inkling/bench_native_stateful.cpp). This is promising as feasibility evidence only: it fixes position, cache capacity, layer, block count, width and execution branch and is not a full model runner. More substantial engineering and end-to-end output validation are required before adopting it.

## Pre-materialized attention weights: rejected at smoke stage

The candidate materialized non-contiguous convolution and attention weight views once before execution. It preserved the current drafted tokens on the three smoke cases, but two paired repetitions per prompt yielded no meaningful balanced gain. [Runs](contiguous-weights-smoke.json), [analysis](contiguous-weights-analysis.json). Runtime defaults remain unchanged. Materialization remains necessary for the experimental MLX export to preserve captured constant values.

## Verified prompt lookup: rejected on copy smoke

Two paired runs each for three copy/code prompts. Lookup engaged (3–8 rounds and 18–36 accepted drafts in the first repetition) but generated a different token sequence on the code case; family-balanced throughput was 0.991x. [Runs](ngram-copy-smoke.json), [analysis](ngram-copy-analysis.json). The candidate fails the strict output gate and is not enabled.

## Six-row attention-output projection sweep: inconclusive

Fifteen interleaved, synchronized calls per geometry, using real packed stages and a synthetic input at layers 0, 12, and 23: [results](dense-o-six.json). The `(rps=2, sg=4, threadgroups=80)` candidate matched sampled output and measured 7.3% faster at layer 12 and 2.7% faster at layer 23, but about 5% slower at layer 0. Isolated kernel timing includes host overhead and cannot establish end-to-end gains; the candidate was not promoted.

## In-process compiled stateful target layer: feasibility only

On the same six-token, layer-0, position-64 test with explicit nonempty cache inputs, `mx.compile` matched direct execution for two distinct input sets and all six output arrays. In one 20-sample alternating microbenchmark, direct median was 1.543 ms and compiled median 1.475 ms (4.4% lower), with 4.2 ms for the first compiled call; [data](stateful-compiled-layer.json). The graph still fixes Python-side sequence length, cache capacity, block count, and position. It has not been integrated or tested end-to-end and does not meet the full-model acceptance gate.

## Dynamic-position compilation constraint

A minimal MLX 0.32.4 `mx.compile` experiment passing positions 64, 65, 64, 66 as Python integers traced four times. The current target path constructs kernel metadata from Python position and KV-length integers and uses those lengths for cache slicing. Therefore the fixed-position stateful layer proof cannot simply be decorated and reused across normal generation; dynamic metadata and cache updates need an explicit-state refactor and parity tests across positions and cache boundaries.

## Dynamic-position sliding layer: inconclusive

A benchmark-only compiled layer accepts position and all populated cache arrays as inputs. For positions 0, 1, 32, 64, 96 and 250 in a 256-token allocation, two independent synthetic input sets matched the output and all five cache arrays bit for bit, with one trace. In 20 alternating calls, median direct and compiled times were 1.794 and 1.789 ms; the difference is below measurement noise. [Results](dynamic-layer-boundaries.json), [prototype](../../runtime_mlx/inkling/bench_dynamic_layer.py). An initial boundary attempt reused a mutable MLX array wrapper between reference cases; the corrected benchmark gives each reference cache an independent wrapper. This checks neither allocation growth nor speculative rollback and does not justify integration or a full C++ rewrite.

## Fresh complete reference: accepted as measurement baseline

[baseline-balanced-256-complete.json](baseline-balanced-256-complete.json) contains all 90 planned measured runs: nine math/code/chat prompts, five serial/drafted repetitions and a 256-token cap. References were stable within each execution mode. Serial/drafted family medians were 28.05/41.72, 27.59/38.16 and 26.70/26.21 tok/s. These compare existing modes, not a new gain. The interrupted historical 35-run checkpoint remains separate. The harness now records frozen workload/environment/source metadata, atomic checkpoints and interrupted normal interpreter exits; analysis rejects missing/duplicated runs and invalid throughput.

## Exact complete-prefill retry cache: accepted

[Balanced analysis](exact-cache-balanced-analysis.json): all 90 runs matched the existing drafted reference's token IDs and stopping. Family first-token latency improved 81.47x/73.05x/71.93x; decode ratios were 0.9991/1.0009/0.9937. Peak active memory including fills was effectively unchanged (maximum addition 384 bytes). [Held-out](exact-cache-heldout-analysis.json) completed 30 runs, [histories](exact-cache-histories-analysis.json) completed 18 runs, and all outputs/stopping matched. Only identical rendered prompts hit; extensions/edits use the original full prefill. Snapshots retain first-token output, final taps, target state and the drafter context before final taps are pushed. Independent wrappers, clone isolation, eviction and budget behavior are tested.

The accepted implementation is `ExactPromptCache`, exported from `runtime_mlx.inkling`. Public opt-in: `lamq serve <model> --drafter <drafter> --reasoning-effort none --prompt-cache 1 --prompt-cache-mode exact`. Entry count and a default 512 MiB retained-array budget bound snapshots. Sampling remains outside the snapshot. Existing server defaults and cache authentication/network restrictions remain unchanged.

## Sampled cache validation: accepted after larger rerun

The initial two-pair sampled study matched outputs but math decode ratio 0.964 narrowly failed the regression gate. [Seven-pair rerun](exact-cache-sampled-seven-analysis.json), temperature 0.7 and 64 tokens, matched all 42 measured runs and passed: family decode ratios 1.0014/0.9896/1.0051. Preserve both reports; use the larger rerun for acceptance.

## Eligible LUT staging change: rejected

The full eligibility scan found only layer 1 and layer 6 gate/up staged among 48 sides. Disabling staging matched the tested outputs but six-token throughput ratios were 0.983/0.859 in [lut-staging-eligible.json](lut-staging-eligible.json). The initially sampled layers 0/12/23 were already unstaged, so those no-op timings cannot support tuning. No dispatch change was promoted.

## Standalone loop-weight scratch policy: below standalone gate

The [standalone 16K comparison](loop-weights-16k-standalone.json) matched all 256 generated tokens and stopping across two pairs. Recomputing identical FP16 loop weights reduced active peak from 12.381 to 11.905 GiB, about 3.8%, missing the 10% standalone memory gate. Default retention remains enabled. This policy is used only to support the accepted optional larger exact cache.

## Larger exact cache with scoped M3 Pro memory policy: accepted with reference limits

Public options are `--prompt-cache-mib 1536 --prompt-cache-low-memory` in exact mode. On fast Apple M3 Pro execution, the latter recomputes identical FP16 prefill weights and temporarily caps unused allocator cache at 256 MiB, honors tighter existing limits, clears unused buffers and restores cache/wired limits on completion, error or early close.

The [long study](exact-cache-long-analysis.json) ran five pairs each at 2K/8K/16K, generating 128 tokens across the first KV-growth boundary. All 30 outputs/stopping matched and memory bounds passed. Both arms used a 256 MiB allocator cap. Its initial 2K decode ratio 0.9669 narrowly failed; the original combined report remains false. The [seven-pair 2K rerun](exact-cache-2k-seven-analysis.json) against original allocator settings passed with ratio 1.0025 and first token 4.923 s to 27.454 ms. The [five-pair 8K original-reference comparison](exact-cache-8k-original-reference-analysis.json) passed with ratio 1.0061 and 20.853 s to 42.769 ms.

The five-pair 16K common-cap result was 43.134 s to 36.336 ms, decode ratio 0.9944, retained snapshot about 1,148 MiB, and conservative active-peak-plus-cap bound 12.153 GiB including fills/refills, below the required 12.32 GiB ceiling. This validates caching under the common cap; it does not isolate allocator-cap performance or compare the full configuration against original allocator settings at 16K. See the final report for the complete acceptance/reference table.

## Profiling and memory obstructions: recorded; deeper rewrite deferred

Xcode `xctrace` was unavailable. MLX capture succeeded but duplicated about 9.8 GB of allocated disk space (18.63 GB logical, 2,930 files). [capture-manifest.json](capture-manifest.json) preserves decoded metadata and file sizes; the owned `/private/tmp/inkling-m3-target.gputrace` payload was removed. The manifest cannot replay it. Capture polluted synchronized component timings; only [target-profile-clean.json](target-profile-clean.json) is valid diagnostic evidence. Component synchronization changes scheduling, and CPU self-time includes native/GPU waits, so neither gives normal GPU durations. Together with flat corrected compilation timing, this does not justify full-model integration in this cycle.

Two 16K uncached warmups failed with Metal out-of-memory callbacks before storing snapshots. Failed checkpoints/logs remain, and later standalone/bounded runs succeeded. [startup-memory.json](startup-memory.json) found unused allocator memory of 1.72 GB after loading, 344 MB after draft-head preparation, and zero after clearing, with active bytes unchanged. This does not prove one cause for both failures. System swap was already about 1.5–1.6 GiB; no recorded thermal warning does not prove temperature stability. Ordinary sandbox execution lacked Metal access, so GPU runs used approved execution.

## Release verification: passed

[release-checks.json](release-checks.json) records zero exits for final Inkling/runtime/fast-kernel tests, server tests, three CLI CUDA tests and the live optional-cache HTTP test. [Ordinary HTTP](exact-cache-live-server.json) and [low-memory HTTP](exact-cache-low-live-server.json) retries returned identical choices/stopping and 25 cached prompt tokens. Eight model-free analysis tests passed. Final documentation, reproduction commands and rollback details are in the [results report](../../docs/inkling-m3-results.md) and [completion log](../../docs/inkling-m3-completion-log.md). No new decode optimization is enabled.

Final cleanup review reproduced a failure to restore temporary memory limits when statistics reporting raised; [before-fix record](cleanup-regression-before-fix.json). Both generators now restore limits around reporting and pending evaluation, including synchronization failures. Regression tests cover reporting in serial/drafted streams and serial pending-evaluation errors. The full runtime/kernel suite passed again after this teardown fix; `release-inkling-cleanup-tests.log` and the appended release-check entry preserve the result. Syntax/whitespace/local-link checks passed, and no owned benchmark/test server remains active.
