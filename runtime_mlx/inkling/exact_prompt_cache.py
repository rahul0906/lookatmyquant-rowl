"""Bounded full-prompt snapshots that preserve the uncached prefill execution shape."""
from __future__ import annotations

import os
import mlx.core as mx

from .generate import PromptCache, _prefill_prompt


class ExactPromptCache(PromptCache):
    """Reuse an identical prompt, including its first-token output and final taps.

    Extensions and edited histories run a fresh prefill. The retained snapshot
    budget defaults to 512 MiB; a larger snapshot is returned without storing it.
    Sampling still runs outside the cache, so a hit does not reuse random draws.
    Use one instance on the same execution thread as its model and drafter.
    """

    def __init__(self, max_entries=1, min_tokens=0, max_bytes=512 * 1024**2, low_memory=False):
        super().__init__(max_entries, min_tokens)
        if max_bytes is not None and max_bytes < 0:
            raise ValueError('max_bytes must be nonnegative or None')
        self.max_bytes = max_bytes
        self.low_memory = low_memory

    @property
    def retained_bytes(self):
        return sum(entry['bytes'] for entry in self._entries)

    def generation_cache_limit(self, m):
        cfg = getattr(m, 'cfg', None)
        if (self.low_memory and getattr(cfg, 'use_fast_metal', False)
                and mx.device_info().get('device_name') == 'Apple M3 Pro'):
            return 256 * 1024**2
        return None

    def prefill(self, m, prompt_ids, *, ordered, prefill_chunk=-1,
                tap_visits=(), drafter=None):
        ids = list(prompt_ids)
        cfg = getattr(m, 'cfg', None)
        low_memory = bool(self.low_memory and getattr(cfg, 'use_fast_metal', False)
                          and mx.device_info().get('device_name') == 'Apple M3 Pro')
        if self.max_entries <= 0 or self.max_bytes == 0:
            self.clear()
            return _prefill_prompt(m, ids, ordered=ordered, prefill_chunk=prefill_chunk,
                                   prompt_cache=None, tap_visits=tap_visits, drafter=drafter,
                                   retain_loop_weights=not low_memory)
        owner = (m, drafter)
        if self._owner is None or any(a is not b for a, b in zip(owner, self._owner)):
            self.clear()
            self._owner = owner
        execution_settings = tuple(os.environ.get(name) for name in (
            'LAMQ_FUSED_SDPA', 'LAMQ_M3_TUNE', 'LAMQ_M4_TUNE', 'LAMQ_DRAFTER_TRIM_BLEN'))
        key = (tuple(ids), ordered, prefill_chunk, tuple(tap_visits), execution_settings, low_memory)
        for entry in self._entries:
            if entry['key'] == key:
                self._clock += 1
                entry['used'] = self._clock
                if drafter is not None:
                    drafter.restore_context(entry['drafter'])
                return (mx.array(entry['out']), entry['cache'].clone(),
                        None if entry['taps'] is None else mx.array(entry['taps']),
                        entry['taps_start'], len(ids))

        out, cache, taps, taps_start, _ = _prefill_prompt(
            m, ids, ordered=ordered, prefill_chunk=prefill_chunk,
            prompt_cache=None, tap_visits=tap_visits, drafter=drafter, retain_loop_weights=not low_memory)
        if self.max_entries <= 0 or len(ids) < max(1, self.min_tokens):
            return out, cache, taps, taps_start, 0

        # Snapshot the drafter before the caller pushes the final returned taps,
        # exactly matching the uncached handoff. All arrays get separate wrappers.
        mx.eval(out, *([] if taps is None else [taps]))
        state = drafter.context_state() if drafter is not None else None
        arrays = [*cache.state, out, *([] if taps is None else [taps])]
        if state is not None:
            arrays.extend(a for a in (*state['ctx_k'], *state['ctx_v'], state['f_rnn']) if a is not None)
        size = sum(a.nbytes for a in arrays)
        if self.max_bytes is not None and size > self.max_bytes:
            return out, cache, taps, taps_start, 0
        self._clock += 1
        entry = {'key': key, 'cache': cache.clone(), 'drafter': state,
                 'out': mx.array(out), 'taps': None if taps is None else mx.array(taps),
                 'taps_start': taps_start, 'used': self._clock, 'bytes': size}
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e['used'], reverse=True)
        while len(self._entries) > self.max_entries or (self.max_bytes is not None and self.retained_bytes > self.max_bytes):
            self._entries.pop()
        return out, cache, taps, taps_start, 0
