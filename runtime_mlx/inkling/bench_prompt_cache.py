"""Replay the recorded long-prompt cache experiment with local cached weights."""
import json,time
from pathlib import Path
import mlx.core as mx
from runtime_mlx import Runtime
from runtime_mlx.runtime import _eos_ids
from runtime_mlx.inkling.generate import PromptCache,stream_dspark
j=json.loads(Path('results/inkling_m3/long-cold.json').read_text())
root=Path.home()/'.cache/huggingface/hub/models--shinmegamiboson--Inkling-2-Mini-LookAtMyQuant-drafter'
draft=root/'snapshots'/(root/'refs/main').read_text().strip()
rt=Runtime.load(Path(j['model']),drafter=draft)
ids=next(r['prompt_ids'] for r in j['runs'] if r['family']=='long_2048')
cache=PromptCache(max_entries=1,min_tokens=256)
res=[]
for name,pc in [('fresh',None),('cache_fill',cache),('cache_hit',cache),('fresh_again',None)]:
 stats={}; t=time.perf_counter(); out=[]
 for chunk in stream_dspark(rt.model,rt.drafter,ids,max_tokens=32,eos_ids=_eos_ids(rt.tokenizer),stats_out=stats,prompt_cache=pc): out.extend(chunk)
 res.append({'arm':name,'tokens':out,'wall_s':time.perf_counter()-t,'stats':stats,'peak_gb':mx.get_peak_memory()/1e9})
 print(json.dumps(res[-1]),flush=True)
Path('results/inkling_m3/prompt-cache.json').write_text(json.dumps(res,indent=2))
