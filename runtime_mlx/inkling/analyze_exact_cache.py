"""Apply the repeated-prefill, parity and memory gates to exact-cache measurements."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

if __package__:
    from .analyze_benchmark import analyze
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location('inkling_pair_analysis', Path(__file__).with_name('analyze_benchmark.py'))
    pair_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair_module)
    analyze = pair_module.analyze


def summarize(data):
    result = analyze(data)
    families = defaultdict(list)
    cases = []
    invalid = []
    for case in data.get('expected_cases', []):
        prompt = case['prompt']
        base = {r['repetition']: r for r in data['runs'] if r['prompt'] == prompt and r['arm'] == 'baseline'}
        cand = {r['repetition']: r for r in data['runs'] if r['prompt'] == prompt and r['arm'] == 'candidate'}
        if not base or base.keys() != cand.keys(): continue
        if any(r['first_token_s'] is None or not math.isfinite(r['first_token_s']) or r['first_token_s'] <= 0
               for r in [*base.values(), *cand.values()]):
            invalid.append(prompt)
            continue
        ratios = [base[i]['first_token_s']/cand[i]['first_token_s'] for i in base]
        ratio = statistics.median(ratios)
        families[case['family']].append(ratio)
        standalone = next(r for r in data['standalone_baselines'] if r['prompt'] == prompt)
        fill = next(r for r in data['cache_fills'] if r['prompt'] == prompt)
        fills = [r for r in data.get('cache_refills', []) if r['prompt'] == prompt]
        def peak_of(row):
            return row.get('peak_reserved_bound_bytes') or row['peak_mlx_bytes']
        peak = max(peak_of(fill), *(peak_of(r) for r in cand.values()), *(peak_of(r) for r in fills))
        cases.append({'family': case['family'], 'prompt': prompt,
            'baseline_first_ms': statistics.median(r['first_token_s'] for r in base.values())*1000,
            'cached_first_ms': statistics.median(r['first_token_s'] for r in cand.values())*1000,
            'first_token_ratio': ratio, 'minimum_paired_ratio': min(ratios),
            'fill_first_ms': fill['first_token_s']*1000,
            'wall_speed_ratio': statistics.median(base[i]['wall_s']/cand[i]['wall_s'] for i in base),
            'peak_bytes_including_fill': peak, 'peak_added_bytes': peak-peak_of(standalone),
            'retained_bytes': max(r['retained_bytes'] for r in cand.values()),
            'all_hits': all(r['stats']['prompt_cached_tokens'] == len(case['prompt_ids']) for r in cand.values())})
    family_ratios = {f: math.exp(statistics.mean(math.log(x) for x in ratios)) for f, ratios in families.items()}
    overall = math.exp(statistics.mean(math.log(x) for x in family_ratios.values())) if family_ratios else None
    headroom_limit = data['device']['max_recommended_working_set_size'] - 1024**3
    result.update(first_token_family_ratios=family_ratios, first_token_balanced_ratio=overall, cases=cases,
        invalid_first_token_cases=invalid,
        passes_repeated_prefill_gate=bool(cases and not invalid and len(cases) == len(data.get('expected_cases', []))
            and result['comparison_complete'] and result['all_candidate_outputs_match_baseline']
            and result['all_baselines_stable'] and overall >= 1.1
            and min(result['families'].values()) >= 0.97
            and all(c['all_hits'] and c['minimum_paired_ratio'] >= 1.1
                    and c['peak_added_bytes'] <= 512*1024**2
                    and c['peak_bytes_including_fill'] <= headroom_limit for c in cases)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    text = json.dumps(summarize(json.loads(a.results.read_text())), indent=2)
    if a.out: a.out.write_text(text + '\n')
    print(text)


if __name__ == '__main__': main()
