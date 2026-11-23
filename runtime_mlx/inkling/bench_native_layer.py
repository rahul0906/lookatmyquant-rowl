"""Export one real target layer and its cache updates for a C++ feasibility study."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--dir',type=Path,required=True)
    a=p.parse_args()
    if not a.model.is_dir():p.error('local model directory required')
    import mlx.core as mx
    from runtime_mlx import Runtime
    m=Runtime.load(a.model).model.inner
    layer=m.layers[0]
    def forward(h):
        cache=m.make_cache()
        result=layer.forward_fast(h,None,0,0,6,cache,0)
        return [result,*cache.state]
    mx.random.seed(687)
    h=mx.random.normal((6,m.cfg.hidden_size)).astype(mx.float32)
    ref=forward(h)
    mx.eval(h,*ref)
    for name in ('k_conv_w', 'v_conv_w', 'post_attn_conv_w', 'post_mlp_conv_w', 'rel_proj_t_f16', 'sink_w_f16'):
        value=mx.contiguous(getattr(layer,name))
        mx.eval(value)
        setattr(layer,name,value)
    ref2=forward(h)
    mx.eval(*ref2)
    for i,(x,y) in enumerate(zip(ref,ref2)):
        print('repeat',i,'max_abs',float(mx.max(mx.abs(x.astype(mx.float32)-y.astype(mx.float32)))),flush=True)
    a.dir.mkdir(parents=True,exist_ok=True)
    mx.save_safetensors(str(a.dir/'layer_inputs.safetensors'),{'hidden':h,**{f'expected_{i}':x for i,x in enumerate(ref)}})
    start=time.perf_counter()
    mx.export_function(str(a.dir/'layer.mlxfn'),forward,h)
    export_s=time.perf_counter()-start
    graph=mx.import_function(str(a.dir/'layer.mlxfn'))
    out=graph(h)
    mx.eval(*out)
    equal=[bool(mx.array_equal(x,y).item()) for x,y in zip(out,ref)]
    if len(out)!=len(ref) or not all(equal):
        for i,(actual,expected) in enumerate(zip(out,ref)):
            diff=mx.max(mx.abs(actual.astype(mx.float32)-expected.astype(mx.float32)))
            print('output',i,'shape',actual.shape,'max_abs',float(diff),flush=True)
        raise RuntimeError(f'export output mismatch {equal}')
    for _ in range(4):
        mx.eval(graph(h));mx.eval(forward(h))
    samples=[]
    direct=[]
    for i in range(20):
        for name in (('direct','graph') if i % 2 == 0 else ('graph','direct')):
            start=time.perf_counter();mx.eval(forward(h) if name == 'direct' else graph(h));elapsed=(time.perf_counter()-start)*1000
            (direct if name == 'direct' else samples).append(elapsed)
    data={'bytes':(a.dir/'layer.mlxfn').stat().st_size,'export_s':export_s,'outputs':len(out),'equal':equal,'median_ms':statistics.median(samples),'samples_ms':samples,'direct_median_ms':statistics.median(direct),'direct_samples_ms':direct}
    (a.dir/'layer-python.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)

if __name__=='__main__':main()
