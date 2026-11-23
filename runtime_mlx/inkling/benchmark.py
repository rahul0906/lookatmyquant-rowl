"""Offline, reproducible Inkling baseline and drafter comparison.

Run with local paths: python -m runtime_mlx.inkling.benchmark MODEL DRAFTER --out results/inkling_m3/run.json
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

PROMPTS = {
    "math": ["What is 17 times 23? Show the steps.", "Solve 3x + 4y = 31 and 5x - 2y = 17 and check your answer.", "A train covers 246 km in 3 hours 25 minutes including a 12 minute stop. Find its moving average speed."],
    "code": ["Write a Python function that merges overlapping intervals with tests.", "Debug this Python function and explain the correction: def average(x): return sum(x) / (len(x)-1)", "Implement binary search in Python and explain its edge cases."],
    "chat": ["Explain how a refrigerator works to a beginner.", "Summarize three causes of the First World War.", "Give a structured checklist for planning a three day trip."],
}
HELDOUT = {
    "math": ["A shop discounts an item by 20 percent and then adds 10 percent tax. What is its final price if it began at 150?"],
    "code": ["Write a Python function to find the first repeated element and include edge cases."],
    "chat": ["Explain the water cycle clearly for a high-school student."],
}
COPY_PROMPTS = {
    "copy": [
        "Repeat this exact text three times: A careful reader compares each result with the original before making a change.",
        "Copy the following code exactly and then explain it:\n\ndef merge(a, b):\n    return sorted(a + b)\n",
        "Quote the following sentence verbatim before summarizing it: The observatory recorded a faint signal on Tuesday and repeated the measurement on Wednesday.",
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("drafter", type=Path)
    ap.add_argument("--candidate-drafter", type=Path)
    ap.add_argument("--candidate-moe-rps", type=int, choices=(2, 4), help="experimental grouped MoE rows per group")
    ap.add_argument("--candidate-core-async", choices=("off",), help="experiment with target-forward graph scheduling")
    ap.add_argument("--candidate-drafter-trim", action="store_true", help="run only the requested drafter block rows")
    ap.add_argument("--candidate-prefill-chunk", type=int, help="compare a positive prefill chunk size to automatic chunking")
    ap.add_argument("--candidate-head-launch", action="store_true", help="experiment with the six-row packed head launch")
    ap.add_argument("--candidate-eval-stride", type=int, choices=(2, 8), help="experiment with target layer submission frequency")
    ap.add_argument("--candidate-compiled-draft", action="store_true", help="experiment with compiled greedy drafter steps")
    ap.add_argument("--candidate-fused-sdpa", action="store_true", help="experiment with thresholded fused attention")
    ap.add_argument("--candidate-qkv-launch", action="store_true", help="experiment with six-row QKV launch geometry")
    ap.add_argument("--candidate-kv-growth", action="store_true", help="experiment with geometric FP16 cache growth")
    ap.add_argument("--candidate-contiguous-weights", action="store_true", help="materialize attention weight views once")
    ap.add_argument("--candidate-ngram", action="store_true", help="experiment with verified prompt lookup drafting")
    ap.add_argument("--candidate-lut-staging", choices=('off',), help="experiment with device-resident MoE lookup tables")
    ap.add_argument("--candidate-loop-weights", choices=('off',), help="experiment without retaining dequantized loop weights")
    ap.add_argument("--long-target", type=int, choices=(2048, 8192, 16384), help="run one long-context size")
    ap.add_argument("--suite", choices=("smoke", "balanced", "heldout", "long", "copy"), default="balanced")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--skip-warmup", action="store_true", help="measure cold long-context runs without additional warmup")
    ap.add_argument("--skip-serial", action="store_true", help="compare drafted arms only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if sum(bool(x) for x in (args.candidate_drafter, args.candidate_moe_rps, args.candidate_core_async, args.candidate_drafter_trim, args.candidate_prefill_chunk, args.candidate_head_launch, args.candidate_eval_stride, args.candidate_compiled_draft, args.candidate_fused_sdpa, args.candidate_qkv_launch, args.candidate_kv_growth, args.candidate_contiguous_weights, args.candidate_ngram, args.candidate_lut_staging, args.candidate_loop_weights)) > 1:
        ap.error("test one candidate at a time")
    if args.candidate_prefill_chunk is not None and args.candidate_prefill_chunk <= 0:
        ap.error("candidate prefill chunk must be positive")
    if args.repetitions < 1 or args.max_tokens < 2:
        ap.error("repetitions must be positive and max-tokens must be at least 2")
    if not args.model.is_dir() or not args.drafter.exists() or (args.candidate_drafter and not args.candidate_drafter.exists()):
        ap.error("model and drafter paths must exist locally")
    os.environ["HF_HUB_OFFLINE"] = "1"
    import mlx.core as mx
    from runtime_mlx import Runtime
    from runtime_mlx.runtime import encode_messages
    from runtime_mlx.inkling.drafter import DSparkDrafterMLX, resolve_drafter_files
    from runtime_mlx.inkling import fast_kernels

    t0 = time.perf_counter()
    rt = Runtime.load(args.model, drafter=args.drafter)
    load_s = time.perf_counter() - t0
    base_weights, cfg = resolve_drafter_files(args.drafter, args.model / "config.json")
    drafters = {"baseline": rt.drafter}
    if args.candidate_drafter:
        candidate_weights, candidate_cfg = resolve_drafter_files(args.candidate_drafter, args.model / "config.json")
        drafters["candidate"] = DSparkDrafterMLX(candidate_weights, candidate_cfg)
    else:
        candidate_weights = None
    if args.candidate_moe_rps or args.candidate_core_async or args.candidate_drafter_trim or args.candidate_prefill_chunk or args.candidate_head_launch or args.candidate_eval_stride or args.candidate_compiled_draft or args.candidate_fused_sdpa or args.candidate_qkv_launch or args.candidate_kv_growth or args.candidate_contiguous_weights or args.candidate_ngram or args.candidate_lut_staging:
        drafters["candidate"] = rt.drafter
    lut_views = [(side, side._stage_rtr, side._lut_args)
                 for layer in rt.model.inner.layers
                 for side in (layer.moe.gu_unified, layer.moe.dn_unified)] if args.candidate_lut_staging else []
    if args.candidate_loop_weights:
        import inspect
        import textwrap
        import types
        from runtime_mlx.inkling import model as model_module
        target = rt.model.inner
        original_forward = target._forward_core_fast
        source = textwrap.dedent(inspect.getsource(type(target)._forward_core_fast))
        needle = 'cache_loop_w = (T >= 64) and (cfg.loop_passes > 1)'
        if source.count(needle) != 1:
            raise RuntimeError('loop weight cache implementation changed; inspect before running')
        source = source.replace(needle, 'cache_loop_w = False')
        namespace = {}
        exec(compile(source, '<experimental-loop-weights>', 'exec'), model_module.__dict__, namespace)
        candidate_forward = types.MethodType(namespace['_forward_core_fast'], target)
        def loop_weight_forward(*values, **kwargs):
            return (candidate_forward if experimental[0] else original_forward)(*values, **kwargs)
        target._forward_core_fast = loop_weight_forward
        drafters['candidate'] = rt.drafter
    weight_views = []
    if args.candidate_contiguous_weights:
        for layer in rt.model.inner.layers:
            for name in ('k_conv_w', 'v_conv_w', 'post_attn_conv_w', 'post_mlp_conv_w', 'rel_proj_t_f16', 'sink_w_f16'):
                original = getattr(layer, name, None)
                if original is not None:
                    materialized = mx.contiguous(original)
                    mx.eval(materialized)
                    weight_views.append((layer, name, original, materialized))
    original_kernel = fast_kernels._make_unified_moe_kernel
    experimental = [False]
    if args.candidate_ngram:
        from runtime_mlx import inkling
        original_stream = inkling.stream_dspark
        def candidate_stream(*values, **kwargs):
            kwargs['ngram'] = experimental[0]
            yield from original_stream(*values, **kwargs)
        inkling.stream_dspark = candidate_stream
    if args.candidate_compiled_draft:
        from runtime_mlx.inkling.drafter import _rnn_update_compiled
        draft = rt.drafter
        original_propose = draft.propose_gpu
        compiled_steps = {}
        def propose_compiled(anchor_tok, embed_fn, head_argmax_fn, draft_len=None):
            if not experimental[0]:
                return original_propose(anchor_tok, embed_fn, head_argmax_fn, draft_len=draft_len)
            count = draft.block_size if draft_len is None else max(2, min(int(draft_len), draft.block_size))
            anchor = anchor_tok.reshape(1).astype(mx.int32) if isinstance(anchor_tok, mx.array) else mx.array([int(anchor_tok)], dtype=mx.int32)
            hidden = draft._run_3layers(embed_fn(anchor)[0, 0, :])
            if count not in compiled_steps:
                def steps(anc, h, f, n=count, head=head_argmax_fn):
                    state = draft._zero_state
                    prev = anc
                    tokens = [anc]
                    for s in range(1, n):
                        emb = draft.markov_w1(prev)[0]
                        inp = mx.concatenate([state, emb, h[s], f], axis=0)[None, :]
                        proj = (draft.rnn_proj_w(inp) + draft.rnn_proj_b)[0]
                        state, latent = _rnn_update_compiled(state, proj)
                        delta = draft.hidden_out_w(latent[None, :])[0] + draft.hidden_out_b
                        prev = head(h[s] + delta).reshape(1).astype(mx.int32)
                        tokens.append(prev)
                    return mx.concatenate(tokens, axis=0)
                compiled_steps[count] = mx.compile(steps)
            return compiled_steps[count](anchor, hidden, draft.f_rnn)
        draft.propose_gpu = propose_compiled
    def candidate_kernel(n, fids, t_mode, rps, sg, **kwargs):
        if experimental[0] and t_mode == "grouped":
            rps = args.candidate_moe_rps
        return original_kernel(n, fids, t_mode, rps, sg, **kwargs)
    if args.candidate_moe_rps:
        fast_kernels._make_unified_moe_kernel = candidate_kernel
    if args.candidate_head_launch:
        stage = rt.model.inner.lm_head.stage
        original_head = stage.matmul_rotated_sg8_f16
        def candidate_head(x, T=8, rb=None, sg=None, ntg=None):
            if experimental[0] and T == 6:
                rb, sg, ntg = 4, 8, 160
            return original_head(x, T=T, rb=rb, sg=sg, ntg=ntg)
        stage.matmul_rotated_sg8_f16 = candidate_head
    if args.candidate_qkv_launch:
        def wrap_qkv(original):
            def qkv(x, T=8, rb=None, sg=None, ntg=None):
                if experimental[0] and T == 6:
                    rb, sg, ntg = 2, 4, 160
                return original(x, T=T, rb=rb, sg=sg, ntg=ntg)
            return qkv
        for layer in rt.model.inner.layers:
            stage = layer.qkv_stage
            stage.matmul_rotated_sg8_f16 = wrap_qkv(stage.matmul_rotated_sg8_f16)
    if args.candidate_core_async:
        target = rt.model.inner
        original_forward = target._forward_core_fast
        def scheduled_forward(*values, **kwargs):
            if not experimental[0]:
                return original_forward(*values, **kwargs)
            original_async = mx.async_eval
            try:
                mx.async_eval = lambda *arrays: None
                return original_forward(*values, **kwargs)
            finally:
                mx.async_eval = original_async
        target._forward_core_fast = scheduled_forward
    if args.candidate_eval_stride:
        import inspect
        import textwrap
        import types
        from runtime_mlx.inkling import model as model_module
        target = rt.model.inner
        original_forward = target._forward_core_fast
        source = textwrap.dedent(inspect.getsource(type(target)._forward_core_fast))
        needle = "(visit & 3) == 0"
        if source.count(needle) != 4:
            raise RuntimeError("target forward changed; inspect scheduling experiment before running")
        source = source.replace(needle, f"(visit & {args.candidate_eval_stride - 1}) == 0")
        namespace = {}
        exec(compile(source, '<experimental-forward>', 'exec'), model_module.__dict__, namespace)
        candidate_forward = types.MethodType(namespace['_forward_core_fast'], target)
        def scheduled_stride_forward(*values, **kwargs):
            return (candidate_forward if experimental[0] else original_forward)(*values, **kwargs)
        target._forward_core_fast = scheduled_stride_forward
    if args.candidate_kv_growth:
        import inspect
        import textwrap
        from runtime_mlx.inkling import model as model_module
        cache_type = model_module.Inkling2Cache
        original_update = cache_type.update_kv_f16
        source = textwrap.dedent(inspect.getsource(original_update))
        needle = "new_cap = ((need + 255) // 256) * 256"
        if source.count(needle) != 1:
            raise RuntimeError("cache growth implementation changed; inspect before running")
        source = source.replace(needle, "new_cap = max(((need + 255) // 256) * 256, ((self.kv_cap[slot] * 5 // 4 + 255) // 256) * 256)")
        namespace = {}
        exec(compile(source, '<experimental-cache-growth>', 'exec'), model_module.__dict__, namespace)
        candidate_update = namespace['update_kv_f16']
        def update_kv(cache, *values, **kwargs):
            return (candidate_update if experimental[0] else original_update)(cache, *values, **kwargs)
        cache_type.update_kv_f16 = update_kv
    source = HELDOUT if args.suite == "heldout" else COPY_PROMPTS if args.suite == "copy" else PROMPTS
    cases = [(family, prompt) for family, prompts in source.items() for prompt in (prompts[:1] if args.suite == "smoke" else prompts)]
    if args.suite == "long":
        paragraphs = (
            "The research team logged measurements, design choices, and observations over several weeks. "
            "Each trial included a setup, a control, a recorded result, and a discussion of uncertainty. "
            "The team reviewed the observations, revised the experiment, and kept the original records.\n"
        )
        cases = []
        for target in ((args.long_target,) if args.long_target else (2048, 8192, 16384)):
            repeated = paragraphs * ((target // 30) + 1)
            content_ids = rt.tokenizer.encode(repeated, add_special_tokens=False)[:target - 100]
            content = rt.tokenizer.decode(content_ids)
            prompt = f"Read the following research notes and summarize the process and limitations.\n\n{content}"
            cases.append((f"long_{target}", prompt))
    encoded = [(family, prompt, encode_messages(rt.tokenizer, [{"role":"user", "content":prompt}])) for family, prompt in cases]
    # Save the entire planned workload before timing. A partial checkpoint must
    # never appear to be a complete comparison simply because later cases have
    # not yet been written.
    expected_cases = [{"family": family, "prompt": prompt, "prompt_ids": ids}
                      for family, prompt, ids in encoded]
    metadata = {
        "model": str(args.model.absolute()), "drafter": str(base_weights.absolute()),
        "candidate_drafter": str(candidate_weights.absolute()) if candidate_weights else None,
        "load_s": load_s, "device": mx.device_info(), "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy")},
        "settings": {**vars(args), "temp": 0, "fused_sdpa": os.environ.get("LAMQ_FUSED_SDPA")},
        "expected_cases": expected_cases,
        "prompt_template_settings": {"reasoning_effort": "none", "add_generation_prompt": True},
        "runtime_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), Path(__file__).parents[1] / 'runtime.py', *[Path(__file__).with_name(name)
                           for name in ('model.py', 'generate.py', 'drafter.py', 'kernels.py', 'fast_kernels.py')]]},
        "git_head": subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    }
    metadata['settings'] = {k: str(v) if isinstance(v, Path) else v
                            for k, v in metadata['settings'].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(status, runs, **extra):
        temporary = args.out.with_suffix(args.out.suffix + '.tmp')
        temporary.write_text(json.dumps({**metadata, 'status': status, 'runs': runs, **extra}, indent=2))
        temporary.replace(args.out)
    checkpoint('running', [])

    def run(ids, drafter, serial=False, candidate=False):
        rt.drafter = drafter
        for side, staged, tables in lut_views:
            side._stage_rtr = False if candidate else staged
            side._lut_args = side._lut_args_mpp if candidate else tables
        for layer, name, original, materialized in weight_views:
            setattr(layer, name, materialized if candidate else original)
        experimental[0] = candidate and bool(args.candidate_moe_rps or args.candidate_core_async or args.candidate_head_launch or args.candidate_eval_stride or args.candidate_compiled_draft or args.candidate_qkv_launch or args.candidate_kv_growth or args.candidate_ngram or args.candidate_loop_weights)
        if args.candidate_drafter_trim:
            os.environ["LAMQ_DRAFTER_TRIM_BLEN"] = "1" if candidate else "0"
        if args.candidate_fused_sdpa:
            os.environ["LAMQ_FUSED_SDPA"] = "1" if candidate else "0"
        mx.reset_peak_memory()
        start = time.perf_counter()
        cpu_start = time.process_time()
        first = None
        previous = start
        gaps = []
        output = []
        last = None
        prefill_chunk = args.candidate_prefill_chunk if candidate and args.candidate_prefill_chunk else -1
        for chunk in rt.stream_tokens(ids, max_tokens=args.max_tokens, use_drafter=not serial, prefill_chunk=prefill_chunk):
            now = time.perf_counter()
            if chunk.token_ids:
                if first is None:
                    first = now
                else:
                    gaps.append(now - previous)
                previous = now
                output.extend(chunk.token_ids)
            last = chunk
        return {"tokens":output, "finish_reason":last.finish_reason, "wall_s":time.perf_counter()-start,
                "process_cpu_s":time.process_time()-cpu_start,
                "first_token_s":None if first is None else first-start,
                "median_stream_gap_s":statistics.median(gaps) if gaps else None,
                "decode_tps":last.generation_tps, "prompt_tps":last.prompt_tps,
                "stats":last.stats, "active_mlx_bytes":mx.get_active_memory(), "peak_mlx_bytes":mx.get_peak_memory(),
                "peak_process_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

    results = []
    completed = [False]
    def mark_interrupted():
        if not completed[0]:
            checkpoint('interrupted', results, obstruction='Process exited before the planned workload completed; consult the command log.')
    atexit.register(mark_interrupted)
    for family, prompt, ids in encoded:
        if not args.skip_warmup:
            for name, d in drafters.items():
                run(ids, d, candidate=name == "candidate")
            if not args.skip_serial:
                run(ids, drafters["baseline"], serial=True)
        for rep in range(args.repetitions):
            names = ([] if args.skip_serial else ["serial"]) + list(drafters)
            if rep % 2:
                names.reverse()
            for name in names:
                row = run(ids, drafters.get(name, drafters["baseline"]), serial=name == "serial", candidate=name == "candidate")
                row.update(family=family, prompt=prompt, prompt_ids=ids, arm=name, repetition=rep)
                results.append(row)
                print(json.dumps({k:row[k] for k in ("family","arm","repetition","decode_tps","first_token_s")}), flush=True)
                checkpoint('running', results)
    grouped = {}
    for family, prompt, ids in encoded:
        rows = [r for r in results if r["prompt"] == prompt]
        refs = [r for r in rows if r["arm"] == "serial"]
        drafted_refs = [r for r in rows if r["arm"] == "baseline"]
        for r in rows:
            r["matches_serial"] = (r["tokens"] == refs[0]["tokens"] and r["finish_reason"] == refs[0]["finish_reason"]) if refs else None
            reference = refs[0] if r["arm"] == "serial" else drafted_refs[0]
            r["matches_baseline"] = r["tokens"] == reference["tokens"] and r["finish_reason"] == reference["finish_reason"]
            r["first_baseline_difference"] = next((i for i, (a, b) in enumerate(zip(r["tokens"], reference["tokens"])) if a != b),
                                                   None if len(r["tokens"]) == len(reference["tokens"]) else min(len(r["tokens"]), len(reference["tokens"])))
        grouped[prompt] = {arm: {"median_decode_tps":statistics.median(r["decode_tps"] for r in rows if r["arm"] == arm),
                                  "matches_serial":all(r["matches_serial"] for r in rows if r["arm"] == arm) if refs else None,
                                  "matches_baseline":all(r["matches_baseline"] for r in rows if r["arm"] == arm)} for arm in (([] if args.skip_serial else ["serial"]) + list(drafters))}
    checkpoint('complete', results, summary=grouped)
    completed[0] = True
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
