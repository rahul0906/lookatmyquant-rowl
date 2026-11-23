"""Replay the recorded code-prompt divergence with different verification widths."""
import json, sys
from pathlib import Path
import mlx.core as mx
import numpy as np
from runtime_mlx import Runtime
from runtime_mlx.inkling.generate import _unwrap_model
r=json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else 'results/inkling_m3/drafter-smoke.json').read_text())
rows=[x for x in r['runs'] if x['family']=='code' and x['repetition']==0]
s=next(x for x in rows if x['arm']=='serial'); d=next(x for x in rows if x['arm']=='baseline')
rt=Runtime.load(Path(r['model']))
m=_unwrap_model(rt.model)
cache=m.make_cache()
_,cache=m.forward_argmax(s['prompt_ids'],cache=cache)
for tok in s['tokens'][:83]:
 out,cache=m.forward_argmax([tok],cache=cache)
 mx.eval(out)
anchor=s['tokens'][83]
print('anchor',anchor,'position',cache.seq_len,flush=True)
c1=cache.clone()
lo,c1=m.forward_ordered_logits([anchor],cache=c1)
mx.eval(lo)
a=np.array(lo[0,-1]); order=np.array(m.lm_head.order_mx)
idx=np.argsort(a)[-5:][::-1]
print('single top',[(int(order[i]),float(a[i])) for i in idx], 'gap',float(a[idx[0]]-a[idx[1]]),flush=True)
for width in (2,4,6,8):
 c=cache.clone()
 ids=[anchor]+d['tokens'][84:84+width-1]
 logits,c=m.forward_all_ordered_logits(ids,cache=c)
 mx.eval(logits)
 b=np.array(logits[0,0]); ix=np.argsort(b)[-5:][::-1]
 print('width',width,'top',[(int(order[i]),float(b[i])) for i in ix], 'max_abs',float(np.max(np.abs(a-b))),flush=True)
