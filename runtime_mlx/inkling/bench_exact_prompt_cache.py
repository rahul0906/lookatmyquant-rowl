"""Measure exact repeated-prompt snapshots using cached local Inkling weights."""
from __future__ import annotations

import argparse
import atexit
import json
import importlib.metadata
import hashlib
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('drafter', type=Path)
    p.add_argument('--suite', choices=('smoke', 'balanced', 'heldout', 'long', 'histories'), default='smoke')
    p.add_argument('--repetitions', type=int, default=5)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--long-target', type=int, choices=(2048, 8192, 16384))
    p.add_argument('--temp', type=float, default=0)
    p.add_argument('--seed', type=int, default=127)
    p.add_argument('--cache-mib', type=int, default=512)
    p.add_argument('--low-memory', action='store_true')
    p.add_argument('--release-for-baseline', action='store_true', help='evict large snapshots before measuring an uncached reference')
    p.add_argument('--allocator-cache-mib', type=int, help='fixed allocator cache budget for both arms')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.repetitions < 1 or a.max_tokens < 2:
        p.error('positive repetitions and at least two generated tokens required')
    import mlx.core as mx
    from runtime_mlx import Runtime
    from runtime_mlx.runtime import _eos_ids, encode_messages
    from .benchmark import PROMPTS, HELDOUT
    from .exact_prompt_cache import ExactPromptCache
    from .generate import stream_dspark
    rt = Runtime.load(a.model, drafter=a.drafter)
    pc = ExactPromptCache(max_entries=1, min_tokens=0, max_bytes=a.cache_mib*1024**2, low_memory=a.low_memory)
    source = HELDOUT if a.suite == 'heldout' else PROMPTS
    cases = [(family, prompt) for family, prompts in source.items()
             for prompt in (prompts[:1] if a.suite == 'smoke' else prompts)]
    if a.suite == 'long':
        old = json.loads(Path('results/inkling_m3/long-cold.json').read_text())
        cases = [(r['family'], r['prompt']) for r in old['runs'] if r['arm'] == 'baseline'
                 and (a.long_target is None or r['family'] == f'long_{a.long_target}')]
    encoded = [(family, prompt, encode_messages(rt.tokenizer, [{'role': 'user', 'content': prompt}]))
               for family, prompt in cases]
    if a.suite == 'histories':
        encoded = []
        replies = {'math': '17 times 23 is 391.',
                   'code': 'Sort intervals by their start, then merge each overlapping interval into the previous one.',
                   'chat': 'A refrigerator moves heat outside by circulating a refrigerant through an evaporator and condenser.'}
        for family, prompts in PROMPTS.items():
            base = [{'role': 'user', 'content': prompts[0]}]
            extended = base + [{'role': 'assistant', 'content': replies[family]},
                               {'role': 'user', 'content': 'Explain another example and check each step.'}]
            edited = [{'role': 'user', 'content': prompts[0] + ' Keep the explanation concise.'}]
            for label, messages in (('original', base), ('extended', extended), ('edited', edited)):
                encoded.append((family, f'{family} history {label}', encode_messages(rt.tokenizer, messages)))
    results = []
    data = {'status': 'running', 'model': str(a.model), 'drafter': str(a.drafter),
            'prompt_template_settings': {'reasoning_effort': 'none', 'add_generation_prompt': True},
            'device': mx.device_info(), 'versions': {p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm', 'numpy')},
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__), Path(__file__).with_name('exact_prompt_cache.py'),
                              Path(__file__).with_name('generate.py'), Path(__file__).with_name('model.py'))},
            'settings': {'repetitions': a.repetitions, 'max_tokens': a.max_tokens,
            'suite': a.suite, 'long_target': a.long_target, 'temp': a.temp, 'seed': a.seed,
            'low_memory': a.low_memory, 'release_for_baseline': a.release_for_baseline,
            'allocator_cache_mib': a.allocator_cache_mib,
            'snapshot_budget_bytes': pc.max_bytes},
            'expected_cases': [{'family': f, 'prompt': p, 'prompt_ids': ids} for f, p, ids in encoded], 'runs': results}
    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, indent=2)); tmp.replace(a.out)
    def mark_interrupted():
        if data['status'] == 'running':
            data.update(status='interrupted', obstruction='Process exited before completing the planned workload; consult the command log.')
            save()
    atexit.register(mark_interrupted)
    def run(ids, cache):
        mx.synchronize()
        old_limit = None
        if a.allocator_cache_mib is not None:
            old_limit = mx.set_cache_limit(a.allocator_cache_mib*1024**2)
        try:
            if old_limit is not None: mx.clear_cache()
            mx.reset_peak_memory()
            mx.random.seed(a.seed)
            stats = {}; tokens = []; first = None; start = time.perf_counter()
            for chunk in stream_dspark(rt.model, rt.drafter, ids, max_tokens=a.max_tokens,
                                      eos_ids=_eos_ids(rt.tokenizer), stats_out=stats, prompt_cache=cache, temp=a.temp):
                if first is None: first = time.perf_counter() - start
                tokens.extend(int(t) for t in chunk)
            mx.synchronize()
            limit = a.allocator_cache_mib*1024**2 if a.allocator_cache_mib is not None else (
                cache.generation_cache_limit(rt.model.inner) if cache is not None else None)
            return {'tokens': tokens, 'finish_reason': 'stop' if tokens[-1] in _eos_ids(rt.tokenizer) else 'length',
                    'wall_s': time.perf_counter()-start, 'first_token_s': first,
                    'decode_tps': stats['decode_tok_s'], 'stats': stats,
                    'peak_mlx_bytes': mx.get_peak_memory(), 'allocator_cache_bytes': mx.get_cache_memory(),
                    'peak_reserved_bound_bytes': mx.get_peak_memory()+limit if limit is not None else None,
                    'retained_bytes': pc.retained_bytes}
        finally:
            if old_limit is not None: mx.set_cache_limit(old_limit)
    save()
    for family, prompt, ids in encoded:
        pc.clear()
        run(ids, None)
        # Timing pairs keep the snapshot resident in both arms. Measure the
        # standalone memory reference separately, before storing any snapshot.
        rt.drafter.reset()
        standalone = run(ids, None)
        data.setdefault('standalone_baselines', []).append({'family': family, 'prompt': prompt, **standalone})
        fill = run(ids, pc)
        data.setdefault('cache_fills', []).append({'family': family, 'prompt': prompt, **fill})
        save()
        for rep in range(a.repetitions):
            for arm in (('baseline', 'candidate') if rep % 2 == 0 else ('candidate', 'baseline')):
                if a.release_for_baseline:
                    if arm == 'baseline':
                        pc.clear(); rt.drafter.reset(); mx.synchronize(); mx.clear_cache()
                    elif not len(pc):
                        refill = run(ids, pc)
                        data.setdefault('cache_refills', []).append({'family': family, 'prompt': prompt, **refill})
                row = run(ids, pc if arm == 'candidate' else None)
                row.update(family=family, prompt=prompt, arm=arm, repetition=rep)
                results.append(row); save()
                print(json.dumps({k: row[k] for k in ('family', 'arm', 'repetition', 'first_token_s', 'decode_tps', 'peak_mlx_bytes')}), flush=True)
                if (row['tokens'] != fill['tokens'] or row['finish_reason'] != fill['finish_reason']
                        or row['tokens'] != standalone['tokens'] or row['finish_reason'] != standalone['finish_reason']):
                    data['status'] = 'rejected_parity'; save()
                    raise RuntimeError(f'output mismatch for {family}, {arm}, repetition {rep}')
    data['status'] = 'complete'; save()


if __name__ == '__main__':
    main()
