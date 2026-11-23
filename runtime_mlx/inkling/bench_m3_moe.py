"""Compare Metal launch geometries on real packed Inkling MoE weights.

Run offline: python -m runtime_mlx.inkling.bench_m3_moe MODEL --out FILE
This microbenchmark does not change runtime dispatch.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=9)
    ap.add_argument('--width', type=int, choices=(1, 6), default=1)
    args = ap.parse_args()
    if not args.model.is_dir() or args.rounds < 3:
        ap.error('local model directory and at least three rounds required')
    import mlx.core as mx
    import numpy as np
    from runtime_mlx import Runtime
    from runtime_mlx.inkling.fast_kernels import _make_unified_moe_kernel, _make_moe_plan_only_kernel, _u32
    rt = Runtime.load(args.model)
    model = rt.model.inner
    mx.random.seed(834)
    cases = []
    for layer in (0, 12, 23):
        for side_name in ('gu_unified', 'dn_unified'):
            side = getattr(model.layers[layer].moe, side_name)
            x_div = 8 if side_name == 'gu_unified' else 1
            n_slots = 8 * args.width
            x = mx.random.normal((n_slots // x_div, side.n_cols)).astype(mx.float16)
            ids = mx.array(([0, 1, 5, 17, 32, 57, 99, 127] * args.width), dtype=mx.uint32)
            inputs_base = [x.reshape(-1), side.codes, side.scales, side._kern_fid, side.exp_code_off, side.exp_scale_off, *side._lut_args, ids]
            nv = [x.reshape(-1), *side._nv_args] if side.has_nvfp4 else []
            configs = [(8,160),(4,40),(4,60),(4,80),(4,120),(4,160),(8,60),(8,80),(8,120),(8,240)]
            if args.width > 1:
                configs = [(sg, ntg, rps) for sg, ntg in configs for rps in (2, 4)]
                plan = _make_moe_plan_only_kernel(side.fids)(inputs=[ids, side.exp_fid, _u32(n_slots)], grid=(128,1,1), threadgroup=(128,1,1), output_shapes=[(129,)], output_dtypes=[mx.uint32])[0]
                inputs_base[-1] = plan
            else:
                configs = [(sg, ntg, 4) for sg, ntg in configs]
            def invoke(sg, ntg, rps):
                mode = 't1' if args.width == 1 else 'grouped'
                kernel = _make_unified_moe_kernel(side.n_cols, side.fids, mode, rps=rps, sg=sg, lpr=None if mode == 't1' else 8, n_rg=side.n_rg, stage_rtr=side._stage_rtr)
                meta = _u32(side.m_rows,n_slots,x_div,ntg) if mode == 't1' else _u32(side.m_rows,x_div,ntg)
                return kernel(inputs=[*inputs_base, meta, *nv], grid=(ntg*sg*32,1,1), threadgroup=(sg*32,1,1), output_shapes=[(n_slots,side.m_rows)], output_dtypes=[mx.float32])[0]
            ref = invoke(8,160,4)
            mx.eval(ref)
            expected = np.array(ref)
            samples = {c: [] for c in configs}
            differences = {}
            # Compile every variant before measuring, then alternate execution order.
            for sg, ntg, rps in configs:
                out = invoke(sg, ntg, rps)
                mx.eval(out)
                differences[(sg, ntg, rps)] = float(np.max(np.abs(np.array(out) - expected)))
            for iteration in range(args.rounds):
                sequence = configs if iteration % 2 == 0 else list(reversed(configs))
                for sg, ntg, rps in sequence:
                    start = time.perf_counter()
                    mx.eval(invoke(sg, ntg, rps))
                    samples[(sg, ntg, rps)].append((time.perf_counter() - start) * 1000)
            for sg, ntg, rps in configs:
                values = samples[(sg, ntg, rps)]
                row = {'layer': layer, 'side': side_name, 'sg': sg, 'threadgroups': ntg, 'rows_per_group': rps,
                       'max_abs_difference': differences[(sg, ntg, rps)], 'samples_ms': values,
                       'median_ms': statistics.median(values)}
                cases.append(row)
                print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'device':mx.device_info(),'model':str(args.model.resolve()),'rounds':args.rounds,'width':args.width,'cases':cases},indent=2))

if __name__ == '__main__':
    main()
