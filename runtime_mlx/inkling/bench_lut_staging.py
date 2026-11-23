"""Interleave real MoE kernels with staged versus device lookup tables on M3."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=(0, 12, 23))
    a = p.parse_args()
    import mlx.core as mx
    from runtime_mlx import Runtime
    model = Runtime.load(a.model).model.inner
    mx.random.seed(891)
    cases = []
    for layer_id in a.layers:
        for name in ('gu_unified', 'dn_unified'):
            side = getattr(model.layers[layer_id].moe, name)
            staged, tables = side._stage_rtr, side._lut_args
            for width in (1, 6):
                x_div = 8 if name == 'gu_unified' else 1
                ids = mx.array(([0, 1, 5, 17, 32, 57, 99, 127]*width), mx.uint32)
                x = mx.random.normal((8*width//x_div, side.n_cols)).astype(mx.float16)
                mx.eval(x, ids)
                def call(candidate, inputs=x):
                    side._stage_rtr = False if candidate else staged
                    side._lut_args = side._lut_args_mpp if candidate else tables
                    return side(inputs, ids, x_div=x_div, x_raw_f16=inputs)
                ref = call(False); out = call(True); mx.eval(ref, out)
                matches = bool(mx.array_equal(ref, out).item())
                other = mx.random.normal(x.shape).astype(mx.float16); mx.eval(other)
                ref2 = call(False, other); out2 = call(True, other); mx.eval(ref2, out2)
                matches = matches and bool(mx.array_equal(ref2, out2).item())
                if not matches: raise RuntimeError(f'LUT parity failed at layer {layer_id}, {name}, T={width}')
                samples = {'baseline': [], 'candidate': []}
                for i in range(21):
                    for arm in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
                        mx.synchronize(); start=time.perf_counter(); mx.eval(call(arm=='candidate'))
                        samples[arm].append((time.perf_counter()-start)*1000)
                row = {'layer':layer_id,'side':name,'width':width,'original_staged':staged,'matches':matches,
                       'baseline_ms':statistics.median(samples['baseline']),
                       'candidate_ms':statistics.median(samples['candidate']),'samples':samples}
                cases.append(row); print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
            side._stage_rtr, side._lut_args = staged, tables
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'model':str(a.model),'device':mx.device_info(),'cases':cases},indent=2))


if __name__ == '__main__': main()
