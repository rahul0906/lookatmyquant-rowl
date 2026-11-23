"""Sweep the actual packed vocabulary head's six-row verification launch."""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--layer', type=int, choices=(0, 12, 23), help='benchmark layer QKV instead of vocabulary head')
    a = p.parse_args()
    if not a.model.is_dir() or a.rounds < 3:
        p.error('local model directory and at least three rounds required')
    import mlx.core as mx
    import numpy as np
    from runtime_mlx import Runtime
    m = Runtime.load(a.model).model.inner
    stage = m.lm_head.stage if a.layer is None else m.layers[a.layer].qkv_stage
    mx.random.seed(345)
    x = mx.random.normal((stage.n_mats, stage.n, 8)).astype(mx.float16)
    configs = [(rb, sg, ntg) for rb in (1,2,4) for sg in (4,8) for ntg in (40,80,160)]
    def call(config):
        rb, sg, ntg = config
        return stage.matmul_rotated_sg8_f16(x, T=6, rb=rb, sg=sg, ntg=ntg)
    baseline=(2,8,80)
    ref=call(baseline)
    mx.eval(ref)
    ref_np=np.array(ref)
    difference={}
    for c in configs:
        out=call(c)
        mx.eval(out)
        difference[c]=float(np.max(np.abs(np.array(out)-ref_np)))
    samples={c:[] for c in configs}
    for rep in range(a.rounds):
        for c in configs if rep % 2 == 0 else list(reversed(configs)):
            start=time.perf_counter()
            mx.eval(call(c))
            samples[c].append((time.perf_counter()-start)*1000)
    rows=[{'rb':c[0], 'sg':c[1], 'threadgroups':c[2], 'max_abs_difference':difference[c],
           'median_ms':statistics.median(samples[c]), 'samples_ms':samples[c]} for c in configs]
    data={'device':mx.device_info(),'model':str(a.model.resolve()),'baseline':baseline,'rounds':a.rounds,'layer':a.layer,'cases':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2))
    print(json.dumps(rows,indent=2))

if __name__ == '__main__':
    main()
