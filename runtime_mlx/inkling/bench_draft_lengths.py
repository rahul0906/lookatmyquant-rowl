"""Compare fixed DSpark lengths with adaptive decoding on the cached Inkling model."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('drafter', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--max-tokens', type=int, default=128)
    a = p.parse_args()
    if not a.model.is_dir() or not a.drafter.exists() or a.rounds < 1 or a.max_tokens < 2:
        p.error('requires local model and drafter paths, positive rounds, and at least two tokens')
    import mlx.core as mx
    from runtime_mlx import Runtime
    from runtime_mlx.runtime import _eos_ids, encode_messages
    from runtime_mlx.inkling.generate import stream_dspark
    from runtime_mlx.inkling.benchmark import PROMPTS
    rt = Runtime.load(a.model, drafter=a.drafter)
    eos = _eos_ids(rt.tokenizer)
    arms = [None, 2, 4, 6, 8]
    data = []
    def run(ids, width):
        stats = {}
        start = time.perf_counter()
        toks=[]
        for chunk in stream_dspark(rt.model, rt.drafter, ids, max_tokens=a.max_tokens, eos_ids=eos,
                                   draft_len=width, adaptive=width is None, stats_out=stats):
            toks.extend(int(x) for x in chunk)
        return {'tokens':toks, 'wall_s':time.perf_counter()-start, 'stats':stats}
    for family, prompts in PROMPTS.items():
        prompt = prompts[0]
        ids = encode_messages(rt.tokenizer, [{'role':'user','content':prompt}])
        for width in arms:
            run(ids, width)
        for rep in range(a.rounds):
            for width in arms if rep % 2 == 0 else list(reversed(arms)):
                row=run(ids,width)
                row.update(family=family,prompt=prompt,prompt_ids=ids,width=width,repetition=rep)
                data.append(row)
                print(json.dumps({'family':family,'width':width,'round':rep,'tps':row['stats'].get('decode_tok_s')}), flush=True)
                a.out.parent.mkdir(parents=True,exist_ok=True)
                a.out.write_text(json.dumps({'status':'running','runs':data},indent=2))
    summary={}
    for family in PROMPTS:
        rows=[r for r in data if r['family']==family]
        reference=next(r for r in rows if r['width'] is None)['tokens']
        summary[family]={str(width):{'median_tps':statistics.median(r['stats']['decode_tok_s'] for r in rows if r['width']==width),
                                     'matches_adaptive':all(r['tokens']==reference for r in rows if r['width']==width)} for width in arms}
    a.out.write_text(json.dumps({'status':'complete','device':mx.device_info(),'model':str(a.model.resolve()),'drafter':str(a.drafter.resolve()),'summary':summary,'runs':data},indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
