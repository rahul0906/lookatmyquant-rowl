"""Exercise the public exact-cache serving options through two local HTTP requests."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('drafter', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache-mib', type=int, default=512)
    p.add_argument('--low-memory', action='store_true')
    a = p.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = ['.venv/bin/lamq', 'serve', str(a.model), '--drafter', str(a.drafter),
               '--host', '127.0.0.1', '--port', str(port), '--model-id', 'inkling',
               '--reasoning-effort', 'none', '--prompt-cache', '1', '--prompt-cache-mode', 'exact',
               '--prompt-cache-mib', str(a.cache_mib),
               '--max-tokens', '64', '--no-warmup']
    if a.low_memory: command.append('--prompt-cache-low-memory')
    env = dict(os.environ)
    env['HF_HUB_OFFLINE'] = '1'
    # The owned test server listens only on loopback. Keep its cache test
    # independent of any authentication settings for other user-owned servers.
    env.pop('LAMQ_API_KEY', None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with a.out.with_suffix('.server.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None: raise RuntimeError('test server exited; see server log')
                try:
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                    conn.request('GET', '/v1/models')
                    response = conn.getresponse(); response.read(); conn.close()
                    if response.status == 200: break
                except (OSError, http.client.HTTPException): pass
                if time.monotonic() >= deadline: raise TimeoutError('test server did not become ready')
                time.sleep(0.25)
            request = {'model': 'inkling', 'messages': [{'role': 'user', 'content': 'What is 17 times 23? Show the steps.'}],
                       'temperature': 0, 'max_tokens': 64}
            for _ in range(2):
                start = time.perf_counter()
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
                conn.request('POST', '/v1/chat/completions', json.dumps(request), {'Content-Type': 'application/json'})
                response = conn.getresponse(); payload = json.loads(response.read()); conn.close()
                if response.status != 200: raise RuntimeError(f'HTTP {response.status}: {payload}')
                results.append({'wall_s': time.perf_counter()-start, 'response': payload})
            first, second = [r['response'] for r in results]
            if first['choices'] != second['choices']: raise RuntimeError('retry changed content or finish reason')
            reused = second['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
            if reused <= 0: raise RuntimeError('exact mode did not report a cache hit')
            a.out.write_text(json.dumps({'status': 'complete', 'command': command, 'same_choices': True,
                                        'cached_tokens': reused, 'runs': results}, indent=2))
            print('Live exact-cache serving passed; cached tokens:', reused, flush=True)
        finally:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()


if __name__ == '__main__': main()
