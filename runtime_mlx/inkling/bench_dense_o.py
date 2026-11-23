"""Sweep real six-token attention-output projection launch geometries."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=15)
    a=p.parse_args()
    if not a.model.is_dir() or a.rounds<3:p.error('local model and at least 3 rounds required')
    import mlx.core as mx
    import numpy as np
    from runtime_mlx import Runtime
    from runtime_mlx.inkling.fast_kernels import _make_dense_stage_kernel,_u32
    m=Runtime.load(a.model).model.inner
    mx.random.seed(773)
    rows=[]
    for layer in (0,12,23):
        stage=m.layers[layer].o_stage
        x=mx.random.normal((stage.n_mats,6,stage.n)).astype(mx.float16)
        configs=[(rps,sg,ntg) for rps in (1,2,4) for sg in (4,8) for ntg in (40,80,160)]
        def call(c):
            rps,sg,ntg=c
            k=_make_dense_stage_kernel(stage.n,stage.fids,6,rps=rps,sg=sg,stage_rtr=stage._stage_rtr)
            return k(inputs=[x.reshape(-1),stage.codes,stage.scales,stage.levels,stage.dims,*stage._lut_args,_u32(stage.total_out,stage.n_mats,ntg)],grid=(ntg*sg*32,1,1),threadgroup=(sg*32,1,1),output_shapes=[(6,stage.total_out)],output_dtypes=[mx.float32])[0]
        ref=call((1,8,160));mx.eval(ref);ref_np=np.array(ref)
        differences={}
        for c in configs:
            o=call(c);mx.eval(o);differences[c]=float(np.max(np.abs(np.array(o)-ref_np)))
        times={c:[] for c in configs}
        for i in range(a.rounds):
            for c in configs if i%2==0 else list(reversed(configs)):
                t=time.perf_counter();mx.eval(call(c));times[c].append((time.perf_counter()-t)*1000)
        rows.extend({'layer':layer,'rps':c[0],'sg':c[1],'threadgroups':c[2],'max_abs_difference':differences[c],'median_ms':statistics.median(times[c]),'samples_ms':times[c]} for c in configs)
        print('completed layer',layer,flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'device':mx.device_info(),'model':str(a.model.resolve()),'rounds':a.rounds,'cases':rows},indent=2))

if __name__=='__main__':main()
