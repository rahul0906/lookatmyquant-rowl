"""Separate warmed CPU attribution and synchronized target components from benchmarks."""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('drafter', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--capture', type=Path)
    a = p.parse_args()
    import mlx.core as mx
    from runtime_mlx import Runtime
    from runtime_mlx.runtime import encode_messages
    from .benchmark import PROMPTS
    from . import model as mm
    rt = Runtime.load(a.model, drafter=a.drafter)
    profiles = {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for family, prompts in PROMPTS.items():
        ids = encode_messages(rt.tokenizer, [{'role': 'user', 'content': prompts[0]}])
        rt.generate(ids, max_tokens=128)
        profiler = cProfile.Profile()
        profiler.enable(); output = rt.generate(ids, max_tokens=128); profiler.disable()
        path = a.out.with_name(a.out.stem + '-' + family + '.prof')
        profiler.dump_stats(str(path))
        stats = pstats.Stats(profiler)
        top = sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[:20]
        profiles[family] = {'profile': str(path), 'total_s': stats.total_tt,
            'top_self_time': [{'file': Path(k[0]).name, 'line': k[1], 'function': k[2],
                              'calls': v[1], 'self_s': v[2], 'cumulative_s': v[3]} for k, v in top]}
        print(f'CPU profile completed: {family}', flush=True)

    ids = encode_messages(rt.tokenizer, [{'role': 'user', 'content': PROMPTS['math'][0]}])
    cache = rt.model.inner.make_cache()
    _, cache = rt.model.inner.forward_argmax(ids, cache=cache)
    mx.eval(*cache.state)
    block = mx.array(ids[-6:])
    _, cache = rt.model.inner.forward_all_argmax(block, cache=cache)
    mx.eval(*cache.state)
    capture = {'requested': bool(a.capture), 'success': False}
    if a.capture:
        started = False
        try:
            a.capture.parent.mkdir(parents=True, exist_ok=True)
            mx.metal.start_capture(str(a.capture))
            started = True
            result = rt.model.inner.forward_all_argmax(block, cache=cache)
            mx.eval(result[0], *cache.state)
            capture.update(success=True, path=str(a.capture))
        except Exception as error:
            capture['error'] = str(error)
        finally:
            if started:
                mx.metal.stop_capture()

    timings = defaultdict(list)
    restores = []
    def arrays(value):
        if isinstance(value, mx.array): return [value]
        if isinstance(value, (tuple, list)): return [x for v in value for x in arrays(v)]
        return []
    def wrap(obj, name, label):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            mx.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            mx.eval(*arrays(result))
            timings[label].append((time.perf_counter()-start)*1000)
            return result
        setattr(obj, name, measured)
        restores.append((obj, name, original))
    for layer in rt.model.inner.layers:
        wrap(layer.qkv_stage, 'matmul_rotated_sg8_f16', 'qkv_projection')
        wrap(layer.qkv_stage, 'matmul_rotated_f16', 'qkv_projection')
        wrap(layer.o_stage, 'matmul_rotated_f16', 'attention_output_projection')
        wrap(layer.moe, 'forward_fast_fused_prerot', 'moe_routing_projections_postconv')
    for name, label in (
        ('gqa_relpos_sinks_sdpa_fast', 'attention'),
        ('block_attn_res_fused', 'block_attention_residual'),
        ('q_rel_clip_fused', 'q_relative_clip'),
        ('kv_sconv4_norm_fused', 'kv_preparation'),
        ('kv_dyn_sconv4_norm_fused', 'dynamic_kv_preparation'),
        ('rht_in_2304_fast', 'qkv_rotation'),
        ('rht_in_3072_gated_fast', 'attention_gate_rotation'),
        ('short_conv4_res_fused', 'post_attention_convolution'),
        ('block_attn_res_norm_rht_fast', 'mlp_input_preparation'),
        ('msg_blend_norm_rht_fast', 'mlp_message_preparation'),
        ('msg_blend_norm_3x_fused', 'attention_message_preparation'),
    ): wrap(mm, name, label)
    try:
        for _ in range(3):
            result = rt.model.inner.forward_all_argmax(block, cache=cache)
            mx.eval(result[0], *cache.state)
    finally:
        for obj, name, original in restores: setattr(obj, name, original)
    result = {'model': str(a.model), 'device': mx.device_info(), 'cpu_profiles': profiles,
              'capture': capture,
              'component_scope': 'Three six-token full-target calls after prefill; synchronization at each component changes scheduling. These are diagnostic host-plus-device call times, not isolated GPU durations or a normal-generation speed estimate.',
              'components': {k: {'calls': len(v), 'total_ms': sum(v), 'mean_ms': sum(v)/len(v)}
                             for k, v in sorted(timings.items(), key=lambda item: sum(item[1]), reverse=True)}}
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['components'], indent=2), flush=True)


if __name__ == '__main__':
    main()
