"""Summarize paired Inkling benchmark runs without requiring MLX."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(data, baseline_arm='baseline', candidate_arm='candidate'):
    runs = data['runs']
    groups = defaultdict(list)
    for row in runs:
        groups[row['prompt']].append(row)
    families = defaultdict(list)
    details = []
    incomplete = []
    for prompt, rows in groups.items():
        baseline_rows = [r for r in rows if r['arm'] == baseline_arm]
        candidate_rows = [r for r in rows if r['arm'] == candidate_arm]
        baseline = {r['repetition']: r for r in baseline_rows}
        candidate = {r['repetition']: r for r in candidate_rows}
        if (not baseline or not candidate or baseline.keys() != candidate.keys()
                or len(baseline) != len(baseline_rows) or len(candidate) != len(candidate_rows)):
            incomplete.append(prompt)
            continue
        reps = sorted(baseline)
        if any(not math.isfinite(r['decode_tps']) or r['decode_tps'] <= 0
               for r in baseline_rows + candidate_rows):
            incomplete.append(prompt)
            continue
        ratios = [candidate[i]['decode_tps'] / baseline[i]['decode_tps'] for i in reps]
        ref = baseline[reps[0]]
        parity = all(candidate[i]['tokens'] == ref['tokens'] and candidate[i]['finish_reason'] == ref['finish_reason'] for i in reps)
        stable = all(baseline[i]['tokens'] == ref['tokens'] and baseline[i]['finish_reason'] == ref['finish_reason'] for i in reps)
        family = ref['family']
        if ratios:
            ratio = statistics.median(ratios)
            families[family].append(ratio)
            details.append({'family': family, 'prompt': prompt, 'paired_ratios': ratios, 'median_ratio': ratio,
                            'candidate_matches_baseline': parity, 'baseline_stable': stable})
    family_ratios = {k: math.exp(statistics.mean(math.log(x) for x in v)) for k, v in families.items()}
    combined = math.exp(statistics.mean(math.log(x) for x in family_ratios.values())) if family_ratios else None
    expected = {case['prompt'] for case in data.get('expected_cases', [])}
    missing = sorted(expected - groups.keys())
    extra = sorted(groups.keys() - expected) if expected else []
    repetitions = data.get('settings', {}).get('repetitions')
    if repetitions is not None:
        for prompt, rows in groups.items():
            for arm in (baseline_arm, candidate_arm):
                if {r['repetition'] for r in rows if r['arm'] == arm} != set(range(repetitions)):
                    if prompt not in incomplete:
                        incomplete.append(prompt)
    complete = data.get('status') == 'complete' and bool(details) and not incomplete and not missing and not extra
    return {'status': data.get('status'), 'baseline_arm': baseline_arm, 'candidate_arm': candidate_arm,
            'complete_prompts': len(details), 'unpaired_prompts': incomplete,
            'missing_prompts': missing, 'unexpected_prompts': extra,
            'comparison_complete': complete, 'families': family_ratios,
            'family_balanced_ratio': combined, 'all_candidate_outputs_match_baseline': complete and all(x['candidate_matches_baseline'] for x in details),
            'all_baselines_stable': complete and all(x['baseline_stable'] for x in details), 'prompts': details}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--baseline-arm', default='baseline')
    p.add_argument('--candidate-arm', default='candidate')
    a = p.parse_args()
    result = analyze(json.loads(a.results.read_text()), a.baseline_arm, a.candidate_arm)
    text = json.dumps(result, indent=2)
    if a.out:
        a.out.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
