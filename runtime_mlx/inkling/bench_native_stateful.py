"""Export one Inkling layer with explicit nonempty KV and convolution-state inputs."""
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
    for name in ('k_conv_w','v_conv_w','post_attn_conv_w','post_mlp_conv_w','rel_proj_t_f16','sink_w_f16'):
        value=mx.contiguous(getattr(layer,name));mx.eval(value);setattr(layer,name,value)
    def forward(h,kv,k,v,attention,mlp):
        c=m.make_cache()
        c.kv_f16[0]=kv
        c.kv_len[0]=64
        c.kv_cap[0]=256
        c.kv_is_sliding[0]=layer.is_sliding
        c.seq_len=64
        c.conv_states[0]=[k,v,attention,mlp]
        result=layer.forward_fast(h,None,0,64,6,c,0)
        return [result,*c.state]
    mx.random.seed(414)
    args=[mx.random.normal((6,2304)).astype(mx.float32),mx.random.normal((2,8,256,128)).astype(mx.float16),
          mx.random.normal((1,3,1024)).astype(mx.float32),mx.random.normal((1,3,1024)).astype(mx.float32),
          mx.random.normal((1,3,2304)).astype(mx.float32),mx.random.normal((1,3,2304)).astype(mx.float32)]
    ref=forward(*args);mx.eval(*args,*ref)
    a.dir.mkdir(parents=True,exist_ok=True)
    mx.save_safetensors(str(a.dir/'stateful_inputs.safetensors'),{**{f'input_{i}':x for i,x in enumerate(args)},**{f'expected_{i}':x for i,x in enumerate(ref)}})
    t=time.perf_counter();mx.export_function(str(a.dir/'stateful.mlxfn'),forward,*args);export_s=time.perf_counter()-t
    fn=mx.import_function(str(a.dir/'stateful.mlxfn'))
    out=fn(*args);mx.eval(*out)
    equals=[bool(mx.array_equal(x,y).item()) for x,y in zip(out,ref)]
    if len(out)!=len(ref) or not all(equals):
        for i,(x,y) in enumerate(zip(out,ref)):
            print('mismatch',i,'max',float(mx.max(mx.abs(x.astype(mx.float32)-y.astype(mx.float32)))),flush=True)
        raise RuntimeError(f'export mismatch: {equals}')
    other=[mx.random.normal(x.shape).astype(x.dtype) for x in args]
    expected_other=forward(*other)
    actual_other=fn(*other)
    mx.eval(*expected_other,*actual_other)
    equals_other=[bool(mx.array_equal(x,y).item()) for x,y in zip(actual_other,expected_other)]
    if len(actual_other)!=len(expected_other) or not all(equals_other):
        raise RuntimeError(f'export mismatch on second input: {equals_other}')
    compiled=mx.compile(forward)
    t=time.perf_counter();compiled_out=compiled(*args);mx.eval(*compiled_out);compile_first_s=time.perf_counter()-t
    compiled_equal=[bool(mx.array_equal(x,y).item()) for x,y in zip(compiled_out,ref)]
    if len(compiled_out)!=len(ref) or not all(compiled_equal):
        raise RuntimeError(f'compiled output mismatch: {compiled_equal}')
    compiled_other=compiled(*other);mx.eval(*compiled_other)
    compiled_other_equal=[bool(mx.array_equal(x,y).item()) for x,y in zip(compiled_other,expected_other)]
    if len(compiled_other)!=len(expected_other) or not all(compiled_other_equal):
        raise RuntimeError(f'compiled output mismatch on second input: {compiled_other_equal}')
    for _ in range(4):mx.eval(fn(*args));mx.eval(forward(*args))
    replay=[];direct=[];compiled_times=[]
    for i in range(20):
        for name in (('direct','replay','compiled') if i%2==0 else ('compiled','replay','direct')):
            t=time.perf_counter();mx.eval(forward(*args) if name=='direct' else fn(*args) if name=='replay' else compiled(*args));elapsed=(time.perf_counter()-t)*1000
            (direct if name=='direct' else replay if name=='replay' else compiled_times).append(elapsed)
    data={'position':64,'input_count':len(args),'output_count':len(out),'equal':equals,'second_input_equal':equals_other,'compiled_equal':compiled_equal,'compiled_second_input_equal':compiled_other_equal,'compile_first_s':compile_first_s,'bytes':(a.dir/'stateful.mlxfn').stat().st_size,'export_s':export_s,'direct_median_ms':statistics.median(direct),'replay_median_ms':statistics.median(replay),'compiled_median_ms':statistics.median(compiled_times),'direct_samples_ms':direct,'replay_samples_ms':replay,'compiled_samples_ms':compiled_times}
    (a.dir/'stateful-python.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)

if __name__=='__main__':main()
