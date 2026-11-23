"""Export and time the real Inkling greedy drafter recurrence for a native MLX experiment."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('drafter', type=Path)
    p.add_argument('--dir', type=Path, required=True)
    a=p.parse_args()
    if not a.model.is_dir() or not a.drafter.exists():
        p.error('local model and drafter required')
    import mlx.core as mx
    from runtime_mlx import Runtime
    from runtime_mlx.inkling.drafter import _rnn_update_compiled
    rt=Runtime.load(a.model,drafter=a.drafter)
    d=rt.drafter
    head=rt.model.inner.lm_head.get_draft_head_argmax()
    def steps(anchor,h,f):
        state=d._zero_state
        prev=anchor
        tokens=[anchor]
        for i in range(1,6):
            emb=d.markov_w1(prev)[0]
            inp=mx.concatenate([state,emb,h[i],f],axis=0)[None,:]
            proj=(d.rnn_proj_w(inp)+d.rnn_proj_b)[0]
            state,latent=_rnn_update_compiled(state,proj)
            delta=d.hidden_out_w(latent[None,:])[0]+d.hidden_out_b
            prev=head(h[i]+delta).reshape(1).astype(mx.int32)
            tokens.append(prev)
        return mx.concatenate(tokens,axis=0)
    mx.random.seed(471)
    anchor=mx.array([149918],dtype=mx.int32)
    hidden=mx.random.normal((8,d.hidden_size)).astype(mx.float32)
    context=mx.random.normal((d.hidden_size,)).astype(mx.float32)
    result=steps(anchor,hidden,context)
    mx.eval(anchor,hidden,context,result)
    a.dir.mkdir(parents=True,exist_ok=True)
    mx.save_safetensors(str(a.dir/'inputs.safetensors'),{'anchor':anchor,'hidden':hidden,'context':context,'expected':result})
    start=time.perf_counter()
    mx.export_function(str(a.dir/'draft.mlxfn'),steps,anchor,hidden,context)
    export_s=time.perf_counter()-start
    fun=mx.import_function(str(a.dir/'draft.mlxfn'))
    out,=fun(anchor,hidden,context)
    mx.eval(out)
    match=bool(mx.array_equal(out,result).item())
    runs=[]
    for _ in range(4):mx.eval(fun(anchor,hidden,context))
    for _ in range(20):
        t=time.perf_counter();mx.eval(fun(anchor,hidden,context));runs.append((time.perf_counter()-t)*1000)
    meta={'export_s':export_s,'bytes':(a.dir/'draft.mlxfn').stat().st_size,'matches':match,'python_import_median_ms':statistics.median(runs),'samples_ms':runs}
    (a.dir/'python.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    main()
