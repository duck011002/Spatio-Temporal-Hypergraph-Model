import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_jev_nyc import (BudgetClient, MODEL, RESERVE, NYC, partition,
                         payload_for, promote, validate_answer)


class JevSafetyTests(unittest.TestCase):
    def test_promotion_keeps_membership_and_neither_falls_back(self):
        rule = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
        base = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        p = np.array([[.8,.1,.1],[.4,.1,.5],[.6,.3,.1]])
        result, switch = promote(rule,base,p,.7)
        np.testing.assert_array_equal(result,[[1,3,2],[6,5,4],[9,8,7]])
        np.testing.assert_array_equal(switch,[True,False,False])

    def test_response_validation_rejects_invalid_probability(self):
        response = {'model':MODEL,'answers':{'next_place':{
            'choice':'A','probabilities':{'A':.7,'B':.2,'neither':.1}}}}
        np.testing.assert_allclose(validate_answer(response),[.7,.2,.1])
        response['answers']['next_place']['probabilities']['A'] = float('nan')
        with self.assertRaises(ValueError):
            validate_answer(response)

    def test_budget_persists_unknown_charge_and_blocks_before_request(self):
        with tempfile.TemporaryDirectory() as folder:
            first = BudgetClient(folder,cap=RESERVE)
            first.record('interrupted',RESERVE,status='reserved')
            second = BudgetClient(folder,cap=RESERVE)
            self.assertEqual(second.total(),RESERVE)
            with patch.dict('os.environ', {'TYPESAFE_API_KEY':'fake'}):
                with patch('requests.Session.post') as post:
                    with self.assertRaises(RuntimeError):
                        second.query({'model':MODEL},'budget_test')
                    post.assert_not_called()
            self.assertNotIn('fake',Path(folder,'api_ledger.jsonl').read_text())

    def test_user_partitions_are_stable(self):
        self.assertEqual(partition(23),partition(23))
        self.assertEqual(set(partition(i) for i in range(100)),{'selection','calibration','audit'})


class JevDataLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nyc = NYC()
        cls.data = cls.nyc.load('validation')

    def test_archived_baselines_and_conflicts(self):
        d = self.data
        self.assertEqual(len(d['queries']),1400)
        self.assertEqual(len(d['conflicts']),407)
        self.assertEqual(int((d['rule'][:,0]==d['batch'].labels).sum()),379)

    def test_target_and_future_event_cannot_enter_payload(self):
        d = self.data
        i = int(d['conflicts'][0])
        before = self.nyc.make_record('validation',d,i)
        columns = ['PoiId','PoiCategoryId','PoiCategoryName','Latitude','Longitude']
        original = d['queries'].loc[i,columns].copy()
        try:
            d['queries'].loc[i,columns] = [999999,999999,'SECRET_TARGET',-89.,179.]
            after = self.nyc.make_record('validation',d,i)
            self.assertEqual(payload_for(before,'evidence'),payload_for(after,'evidence'))
        finally:
            d['queries'].loc[i,columns] = original
        self.assertLess(before['history_max_epoch'],before['query_epoch'])
        self.assertLessEqual(before['history_max_epoch'],before['previous_epoch'])
        payload = payload_for(before,'evidence')
        self.assertNotIn('user',payload['state'])
        self.assertNotIn('sample_id',payload['state'])
        plain = json.dumps(payload)
        self.assertNotIn('SECRET_TARGET',plain)
        # Perturb all unavailable future events for this user, not only the label row.
        history = self.nyc.histories[before['user']]
        backup = history.copy()
        try:
            history.loc[history.UTCTimeOffsetEpoch>=before['query_epoch'],'PoiCategoryId'] = 999999
            future_changed = self.nyc.make_record('validation',d,i)
            self.assertEqual(payload,payload_for(future_changed,'evidence'))
        finally:
            self.nyc.histories[before['user']] = backup

    def test_option_swap_and_context_score_removal(self):
        r = self.nyc.make_record('validation',self.data,int(self.data['conflicts'][0]))
        regular, swapped = payload_for(r,'evidence'),payload_for(r,'evidence',True)
        self.assertEqual(regular['state']['candidates']['A'],swapped['state']['candidates']['B'])
        self.assertEqual(regular['state']['candidates']['B'],swapped['state']['candidates']['A'])
        contextual = payload_for(r,'context')
        for c in contextual['state']['candidates'].values():
            for key in ['model_score','fused_rule_score','rule_components']:
                self.assertNotIn(key,c)


if __name__ == '__main__':
    unittest.main()
