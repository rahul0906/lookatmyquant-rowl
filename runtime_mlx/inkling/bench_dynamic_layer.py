"""Prototype one dynamically positioned, compiled Inkling sliding-attention layer.

This is an isolated experiment for a six-token block and a fixed-capacity FP16 cache.
"""
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
    a=p.parse_args()
    if not a.model.is_dir():p.error('local model required')
    import mlx.core as mx
    from runtime_mlx import Runtime
    from runtime_mlx.inkling import model as model_module
    from runtime_mlx.inkling.fast_kernels import _make_gqa_relpos_sdpa_kernel
    m=Runtime.load(a.model).model.inner
    layer=m.layers[0]
    if not layer.is_sliding:raise RuntimeError('prototype expects a sliding layer')
    mx.random.seed(127)
    arrays=[mx.random.normal((6,2304)).astype(mx.float32),mx.random.normal((2,8,256,128)).astype(mx.float16),
            mx.random.normal((1,3,1024)).astype(mx.float32),mx.random.normal((1,3,1024)).astype(mx.float32),
            mx.random.normal((1,3,2304)).astype(mx.float32),mx.random.normal((1,3,2304)).astype(mx.float32)]
    mx.eval(*arrays)
    def cache_with(kv,k,v,attn,mlp,pos):
        # Assignment into an MLX array updates its Python wrapper. Give the
        # reference cache its own wrapper so one case cannot alter the next.
        c=m.make_cache();c.kv_f16[0]=mx.array(kv);c.kv_len[0]=pos;c.kv_cap[0]=256;c.kv_is_sliding[0]=True;c.seq_len=pos
        c.conv_states[0]=[k,v,attn,mlp]
        return c
    def reference(h,kv,k,v,attn,mlp,pos):
        c=cache_with(kv,k,v,attn,mlp,pos)
        y=layer.forward_fast(h,None,0,pos,6,c,0)
        return [y,*c.state]
    positions=(0,1,32,64,96,250)
    expected={}
    for pos in positions:
        expected[pos]=reference(*arrays,pos);mx.eval(*expected[pos])
    traces=[]
    def dynamic(h,kv,k,v,attn,mlp,pos):
        traces.append(1)
        c=cache_with(kv,k,v,attn,mlp,64)
        def update(slot,new,is_sliding):
            buf=mx.slice_update(kv,new,pos.reshape(1),axes=[2])
            c.kv_f16[slot]=buf
            return buf,pos+6,0,256,False,None
        c.update_kv_f16=update
        original=model_module.gqa_relpos_sinks_sdpa_fast
        def attention(q,kv_buf,rel,proj,q_in,sink,T,kv_len,kv_start,q_pos0,kv_cap,is_sliding,**kwargs):
            meta=mx.stack([mx.array(6,mx.uint32), (pos+6).astype(mx.uint32),mx.array(0,mx.uint32),pos.astype(mx.uint32),mx.array(256,mx.uint32),mx.array(1,mx.uint32)])
            return _make_gqa_relpos_sdpa_kernel(False,kv_fp8=False)(inputs=[q.reshape(-1),kv_buf.reshape(-1),rel.reshape(-1),proj.reshape(-1),q_in.reshape(-1),sink.reshape(-1),meta],grid=(128,144,1),threadgroup=(128,1,1),output_shapes=[(6,3072)],output_dtypes=[mx.float32])[0]
        model_module.gqa_relpos_sinks_sdpa_fast=attention
        try:
            y=layer.forward_fast(h,None,0,64,6,c,0)
            return [y,*c.state]
        finally:
            model_module.gqa_relpos_sinks_sdpa_fast=original
    compiled=mx.compile(dynamic)
    matches={}
    for pos in positions:
        out=compiled(*arrays,mx.array(pos,mx.uint32));mx.eval(*out)
        matches[pos]=[bool(mx.array_equal(x,y).item()) for x,y in zip(out,expected[pos])]
        if len(out)!=len(expected[pos]) or not all(matches[pos]):
            for i,(x,y) in enumerate(zip(out,expected[pos])):
                print('position',pos,'output',i,'max_abs',float(mx.max(mx.abs(x.astype(mx.float32)-y.astype(mx.float32)))),flush=True)
            raise RuntimeError(f'nonmatching outputs: {matches[pos]}')
    other=[mx.random.normal(x.shape).astype(x.dtype) for x in arrays]
    mx.eval(*other)
    for pos in positions:
        ref=reference(*other,pos); out=compiled(*other,mx.array(pos,mx.uint32));mx.eval(*ref,*out)
        if len(ref)!=len(out) or not all(bool(mx.array_equal(x,y).item()) for x,y in zip(ref,out)):
            raise RuntimeError(f'second input mismatch at position {pos}')
    direct=[];timed=[]
    for i in range(20):
        pos=positions[i%len(positions)];position=mx.array(pos,mx.uint32);mx.eval(position)
        for arm in (('direct','compiled') if i%2==0 else ('compiled','direct')):
            start=time.perf_counter();mx.eval(reference(*arrays,pos) if arm=='direct' else compiled(*arrays,position));ms=(time.perf_counter()-start)*1000
            (direct if arm=='direct' else timed).append(ms)
    result={'positions':positions,'matches':matches,'traces':len(traces),'direct_median_ms':statistics.median(direct),'compiled_median_ms':statistics.median(timed),'direct_samples_ms':direct,'compiled_samples_ms':timed}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
