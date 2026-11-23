# Inkling M3 Pro optimization plan

Status: the measured optimization cycle completed on 23 November 2026. Exact repeated-prompt caching and its optional M3 Pro memory policy passed the applicable cache gates and are available explicitly. No new decode candidate passed the 5% balanced gate. See the [final results](inkling-m3-results.md) and [experiment ledger](../results/inkling_m3/ledger.md) for evidence and dispositions.

Optimize the cached Inkling-2-Mini LookAtMyQuant 8.9 GB pack and its DSpark drafter for M3 Pro. Keep the model weights, FP16 KV cache, tokenizer, reasoning settings, full-target verification, and reference runtime unchanged. Require identical greedy tokens and stopping behavior on the frozen validation suite; this is evidence, not a proof for all inputs.

## Measurement

Freeze model and drafter paths and revisions, rendered token IDs, versions, settings, and serial and drafted reference outputs. Measure math, coding, chat, and long-context workloads, including repeated and edited prefixes. Record cold load, time to first token, prefill, decode, acceptance, memory, and system conditions. Warm execution shapes and use five interleaved baseline/candidate pairs, seven if inconclusive. Profile independently from timing.

Accept a decode change only with at least 5% family-balanced geometric mean improvement, no family regression over 3%, and exact greedy parity. Require 10% improvement for prefill, startup, cache or memory changes, with no decode regression over 3%. Gains must exceed noise. Peak memory may rise by at most 512 MiB and must remain at least 1 GiB below the recommended working set. Stress cases already exceeding that limit must improve memory.

## Experiments

1. Compare the cached model-specific 8.9 GB drafter to the manifest-selected drafter.
2. Compare existing fused attention modes on M3 across decode, verification, and context sizes.
3. Sweep M3-specific dense, MoE, and head launch parameters, independently for decode, verification and prefill.
4. Reduce CPU/GPU synchronization in greedy verification; preserve EOS and cache commit boundaries.
5. Fuse drafter recurrent and shortlist operations if profiling shows worthwhile overhead.
6. Tune speculative block length based on measured cost and acceptance; separately evaluate tree and prompt lookup modes.
7. Fuse target-side operations only where traces show a critical path gain.
8. Evaluate KV allocation, sliding-window compaction, buffer reuse, and prefix snapshots while preserving exact state.
9. Compare prefill chunking and resident execution, including repeated and edited histories.
10. Compile larger pure MLX regions with explicit cache state. Explore C++ only if remaining host overhead justifies it.
11. Evaluate isolated MLX versions, lossless loading caches, and startup compilation separately.

Reject re-quantization, compressed KV, reduced reasoning, approximate attention, skipped model work, acceptance margins, and target-vocabulary pruning. Specialize only for detected M3 Pro geometry, preserving a fallback and `LAMQ_M3_TUNE=0` for default-enabled tuning.

## Validation and deliverables

Check kernel outputs against reference computations, all 48 visit states, cache growth and rollback, EOS, streaming, serial/drafted parity, and sampled verification. Keep a held-out set separate from tuning. Record first divergent token and logits where possible. Run the Inkling and relevant CLI/server tests after changes. Maintain an experiment ledger with accepted, rejected, inconclusive and deferred results. Combine accepted candidates incrementally, retest combinations, and provide a reproducible baseline and final configuration.

## Final disposition

The fresh 90-run balanced serial/drafted baseline, balanced exact-cache comparison, held-out prompts, original/extended/edited histories, seeded sampling, 2K/8K/16K cache tests, and public HTTP checks are complete. Final runtime/kernel, server and CLI tests passed. The acceptance reference for the new cache is the existing drafted path; pre-existing serial/drafted numerical differences were diagnosed and documented.

Launch, attention, drafter, synchronization, loading, KV-growth, LUT and native/compiled experiments did not establish an eligible decode gain. Full-model compilation/C++ integration and further GPU-duration tracing are deferred: corrected dynamic-layer timing was flat, and Xcode trace inspection was unavailable. No new long-context serial-versus-drafted quality assessment was completed at 16K; long-context cache parity was measured against the existing drafted reference. These limits remain documented, rather than being inferred from the cache tests.
