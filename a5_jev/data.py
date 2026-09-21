"""Label-free request construction, aligned to frozen candidate exports."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rule_teacher import RankingBatch

MODEL = 'jev-1.13.0'
PROMPT_VERSION = 'a5-category-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def category_name(value):
    text = str(value)
    if text.startswith('['):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return str(parsed[0].get('name', 'unknown'))
        except (ValueError, SyntaxError):
            pass
    return text if text and text != 'nan' else 'unknown'


def partition(dataset, user):
    bucket = int(digest(['a5-user-partition-v1', dataset, int(user)]), 16) % 10
    return 'selection' if bucket < 6 else 'calibration' if bucket < 8 else 'audit'


class Dataset:
    def __init__(self, dataset, data_dir, candidate_dir, split):
        if dataset not in ('nyc', 'ca', 'tky') or split not in ('validation', 'test'):
            raise ValueError('Invalid dataset or split')
        self.dataset, self.split = dataset, split
        self.data_dir, self.candidate_dir = Path(data_dir), Path(candidate_dir)
        self.events_path = self.data_dir/'sample.csv'
        self.query_path = self.data_dir/('validate_sample.csv' if split == 'validation' else 'test_sample.csv')
        self.candidate_path = self.candidate_dir/f'{split}_candidates.npz'
        events = pd.read_csv(self.events_path, low_memory=False)
        events['CategoryText'] = events.PoiCategoryName.map(category_name)
        train = events[events.SplitTag.str.lower() == 'train']
        self.meta = train.groupby('PoiId').CategoryText.agg(lambda s: s.mode().iloc[0]).to_dict()
        self.history = {int(u): g.sort_values(['UTCTimeOffsetEpoch', 'check_ins_id']) for u, g in events.groupby('UserId')}
        self.empty_history = events.iloc[:0]
        self.batch = RankingBatch.load(str(self.candidate_path))
        queries = pd.read_csv(self.query_path)
        if self.batch.sample_indices is not None:
            queries = queries.iloc[self.batch.sample_indices.astype(int)]
        self.queries = queries.reset_index(drop=True)
        if not np.array_equal(self.batch.labels, self.queries.PoiId.to_numpy()):
            raise ValueError('Candidate labels do not align with queries')
        if self.batch.candidate_ids.shape[1] != 20 or np.any(np.diff(self.batch.model_scores, axis=1) > 0):
            raise ValueError('Expected sorted Top-20 candidate cache')
        self.categories = [[str(self.meta.get(int(p), 'unknown')) for p in row] for row in self.batch.candidate_ids]
        self.partitions = np.array([partition(dataset, u) for u in self.queries.UserId])

    def provenance(self):
        return dict(dataset=self.dataset, split=self.split, n=len(self.queries), prompt_version=PROMPT_VERSION,
                    data_hash=sha256(self.events_path), query_hash=sha256(self.query_path), candidate_hash=sha256(self.candidate_path))

    def request(self, index):
        row = self.queries.iloc[index]
        target, last = int(row.UTCTimeOffsetEpoch), int(row.last_checkin_epoch_time)
        history = self.history.get(int(row.UserId), self.empty_history)
        history = history[(history.UTCTimeOffsetEpoch <= last) & (history.UTCTimeOffsetEpoch < target)]
        names = sorted(set(self.categories[index]))
        seed = int(digest([PROMPT_VERSION, self.dataset, self.split, int(row.check_ins_id)]), 16) % 2**32
        names = np.random.default_rng(seed).permutation(names).tolist()
        stamp = pd.Timestamp(row.UTCTimeOffset)
        state = dict(task='Predict the category of the next actual check-in, not a recommended activity.',
                     query_hour=int(stamp.hour), query_weekday=stamp.day_name(), observed_user_history_length=len(history),
                     recent_visits_oldest_first=[dict(category=e.CategoryText, hours_before_query=round((target-int(e.UTCTimeOffsetEpoch))/3600, 6)) for e in history.tail(12).itertuples()],
                     category_options={f'k{j:02d}': c for j, c in enumerate(names)})
        criteria = {f'k{j:02d}': f'The next actual check-in has category: {c}.' for j, c in enumerate(names)}
        criteria['outside_category'] = 'The next actual check-in has a category absent from the listed options.'
        instructions = ("Infer the next observed category from this person's recent temporal sequence and query time. "
                        'Use habitual context but do not invent an itinerary. Sparse check-ins can omit intermediate activities. '
                        'Options are randomly ordered, with no popularity or ranking information. Different places can share '
                        'a category. outside_category means a different CATEGORY, not a different place of a listed category. '
                        'Distribute probability over plausible alternatives and account for uncertainty.')
        return dict(model=MODEL, state=state, questions={'next_category': dict(type='choice', instructions=instructions, criteria=criteria)}), names
