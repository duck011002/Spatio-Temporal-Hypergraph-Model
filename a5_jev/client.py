"""Bounded, resumable official Jev transport. Credentials never persisted."""
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import requests
from .data import MODEL, canonical, digest, save

PRICE = .042/1_000_000
MIN_RESERVE_TOKENS = 1_536


def validate(payload, response):
    if response.get('model') != MODEL:
        raise ValueError('Unexpected Jev model')
    keys = list(payload['questions']['next_category']['criteria'])
    answer = response['answers']['next_category']
    if set(answer['probabilities']) != set(keys) or answer['choice'] not in keys:
        raise ValueError('Response category schema differs from request')
    p = np.array([answer['probabilities'][k] for k in keys], float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or abs(p.sum()-1) > .12:
        raise ValueError('Invalid Jev probability mass')
    return (p/p.sum()).tolist()


class Client:
    def __init__(self, directory, cap, offline=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cap, self.offline = float(cap), offline
        if not np.isfinite(self.cap) or self.cap <= 0:
            raise ValueError('Budget cap must be positive')
        self.ledger = self.directory/'ledger.jsonl'
        self.lock, self.local = threading.Lock(), threading.local()
        self.stop = threading.Event()
        self.charges = {}
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding='utf-8').splitlines():
                item = json.loads(line)
                self.charges[item['attempt']] = item['charge_usd']

    def total(self):
        with self.lock:
            return sum(self.charges.values())

    def record(self, attempt, charge, **fields):
        item = dict(attempt=attempt, charge_usd=charge, **fields)
        with self.ledger.open('a', encoding='utf-8') as f:
            f.write(canonical(item)+'\n'); f.flush(); os.fsync(f.fileno())
        self.charges[attempt] = charge

    def query(self, payload):
        key = digest(payload)
        path = self.directory/'cache'/f'{key}.json'
        if path.exists():
            entry = json.loads(path.read_text(encoding='utf-8'))
            if entry['request_hash'] != key or digest(entry['request']) != key:
                raise ValueError('Corrupt cache')
            return validate(payload, entry['response'])
        if self.offline:
            raise FileNotFoundError(f'Missing offline response {key}')
        token = os.environ.get('TYPESAFE_API_KEY')
        if not token:
            raise RuntimeError('TYPESAFE_API_KEY is required')
        if len(canonical(payload)) > 32000:
            raise ValueError('Request exceeds bounded input size')
        # A fixed 64k reservation would stop a normal CA/TKY run too early.
        estimated_tokens = max(
            MIN_RESERVE_TOKENS,
            math.ceil(len(canonical(payload).encode('utf-8')) / 4) + 256,
        )
        reserve = estimated_tokens * PRICE
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        for retry in range(3):
            attempt = uuid.uuid4().hex
            with self.lock:
                if self.stop.is_set() or sum(self.charges.values())+reserve > self.cap:
                    self.stop.set(); raise RuntimeError('Budget or failure guard stopped calls')
                self.record(attempt, reserve, status='reserved', request_hash=key,
                            reserved_tokens=estimated_tokens)
            start = time.perf_counter()
            try:
                response = self.local.session.post('https://api.typesafe.ai/v1/systemone',
                    headers={'Authorization': f'Bearer {token}'}, json=payload, timeout=(10,45))
            except requests.RequestException:
                if retry < 2:
                    time.sleep(2**retry); continue
                self.stop.set(); raise RuntimeError('Network retries exhausted; resume from cache') from None
            if response.status_code == 200:
                data = response.json()
                tokens = int(data['usage']['input_tokens'])
                if tokens < 0:
                    self.stop.set(); raise ValueError('Invalid token usage')
                with self.lock:
                    self.record(attempt, tokens*PRICE, status='success', request_hash=key, input_tokens=tokens,
                                latency_s=time.perf_counter()-start)
                p = validate(payload, data)
                save(path, dict(request_hash=key, request=payload, response=data))
                return p
            if response.status_code == 429 or response.status_code >= 500:
                header = response.headers.get('Retry-After')
                try:
                    delay = float(header) if header else 2**retry
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    delay = parsedate_to_datetime(header).timestamp()-time.time()
                if retry < 2 and delay <= 60:
                    time.sleep(max(1,delay)); continue
            self.stop.set()
            raise RuntimeError(f'Jev HTTP {response.status_code}; reserved cost retained')
        raise RuntimeError('Request failed')
