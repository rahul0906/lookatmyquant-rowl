"""Checks for result comparison without needing a model or GPU."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('inkling_benchmark_analysis', Path(__file__).parent / 'inkling' / 'analyze_benchmark.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(family, prompt, arm, repetition, tps, tokens, finish='length'):
    return {'family': family, 'prompt': prompt, 'arm': arm, 'repetition': repetition,
            'decode_tps': tps, 'tokens': tokens, 'finish_reason': finish}


def test_detects_parity_and_uses_paired_family_ratios():
    data = {'status': 'complete', 'runs': [
        row('math', 'a', 'baseline', 0, 10, [1, 2]), row('math', 'a', 'candidate', 0, 12, [1, 2]),
        row('math', 'a', 'baseline', 1, 20, [1, 2]), row('math', 'a', 'candidate', 1, 24, [1, 2]),
        row('code', 'b', 'baseline', 0, 10, [3]), row('code', 'b', 'candidate', 0, 10, [4]),
    ]}
    result = module.analyze(data)
    assert result['complete_prompts'] == 2
    assert result['families']['math'] == 1.2
    assert not result['all_candidate_outputs_match_baseline']
    assert result['all_baselines_stable']


def test_incomplete_pairs_are_not_reported_as_complete():
    data = {'status': 'running', 'runs': [row('math', 'a', 'baseline', 0, 10, [1])]}
    result = module.analyze(data)
    assert result['complete_prompts'] == 0
    assert result['family_balanced_ratio'] is None
    assert not result['all_candidate_outputs_match_baseline']


def test_partial_suite_cannot_pass_on_completed_cases():
    data = {'status': 'complete', 'runs': [
        row('math', 'a', 'baseline', 0, 10, [1]), row('math', 'a', 'candidate', 0, 11, [1]),
        row('code', 'b', 'baseline', 0, 10, [2]),
    ]}
    result = module.analyze(data)
    assert result['complete_prompts'] == 1
    assert result['unpaired_prompts'] == ['b']
    assert not result['comparison_complete']
    assert not result['all_candidate_outputs_match_baseline']


def test_duplicate_repetition_cannot_hide_a_failed_run():
    data = {'status': 'complete', 'runs': [
        row('math', 'a', 'baseline', 0, 10, [1]), row('math', 'a', 'candidate', 0, 11, [2]),
        row('math', 'a', 'candidate', 0, 11, [1]),
    ]}
    assert not module.analyze(data)['comparison_complete']


def test_expected_cases_and_repetition_count_must_be_present():
    data = {'status': 'complete', 'settings': {'repetitions': 2},
            'expected_cases': [{'prompt': 'a'}, {'prompt': 'b'}], 'runs': [
                row('math', 'a', 'baseline', 0, 10, [1]),
                row('math', 'a', 'candidate', 0, 11, [1]),
            ]}
    result = module.analyze(data)
    assert result['missing_prompts'] == ['b']
    assert result['unpaired_prompts'] == ['a']
    assert not result['comparison_complete']


def test_serial_and_drafted_reference_comparison():
    data = {'status': 'complete', 'settings': {'repetitions': 1},
            'expected_cases': [{'prompt': 'a'}], 'runs': [
                row('math', 'a', 'serial', 0, 10, [1]),
                row('math', 'a', 'baseline', 0, 15, [2]),
            ]}
    result = module.analyze(data, 'serial', 'baseline')
    assert result['comparison_complete']
    assert result['family_balanced_ratio'] == 1.5
    assert result['all_baselines_stable']
    assert not result['all_candidate_outputs_match_baseline']


def test_invalid_throughput_cannot_pass_or_crash_analysis():
    for value in (0, -1, float('nan'), float('inf')):
        data = {'status': 'complete', 'runs': [
            row('math', 'a', 'baseline', 0, 10, [1]),
            row('math', 'a', 'candidate', 0, value, [1]),
        ]}
        result = module.analyze(data)
        assert not result['comparison_complete']
        assert result['family_balanced_ratio'] is None


def test_exact_cache_gate_rejects_memory_parity_missing_hits_and_regressions():
    import copy
    spec = importlib.util.spec_from_file_location('exact_cache_analysis', Path(__file__).parent / 'inkling' / 'analyze_exact_cache.py')
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    base = row('math', 'a', 'baseline', 0, 10, [1])
    cand = row('math', 'a', 'candidate', 0, 10, [1])
    for r, first in ((base, 1), (cand, 0.1)):
        r.update(first_token_s=first, wall_s=2, peak_mlx_bytes=1024, retained_bytes=512,
                 stats={'prompt_cached_tokens': 1})
    data = {'status': 'complete', 'settings': {'repetitions': 1},
            'device': {'max_recommended_working_set_size': 4*1024**3},
            'expected_cases': [{'family': 'math', 'prompt': 'a', 'prompt_ids': [1]}],
            'runs': [base, cand], 'standalone_baselines': [copy.deepcopy(base)],
            'cache_fills': [copy.deepcopy(base)]}
    assert exact.summarize(data)['passes_repeated_prefill_gate']
    for field, value in (('decode_tps', 9.6), ('tokens', [2]), ('first_token_s', None),
                         ('first_token_s', float('nan')), ('peak_mlx_bytes', 4*1024**3)):
        changed = copy.deepcopy(data)
        changed['runs'][1][field] = value
        assert not exact.summarize(changed)['passes_repeated_prefill_gate']
    changed = copy.deepcopy(data)
    changed['runs'][1]['stats']['prompt_cached_tokens'] = 0
    assert not exact.summarize(changed)['passes_repeated_prefill_gate']
    changed = copy.deepcopy(data)
    changed['cache_fills'][0]['peak_mlx_bytes'] = 600*1024**2
    assert not exact.summarize(changed)['passes_repeated_prefill_gate']
