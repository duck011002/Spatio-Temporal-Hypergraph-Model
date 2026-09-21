import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from a5_jev.core import fuse
from a5_jev.data import Dataset, MODEL, digest, save, sha256
from a5_jev.client import Client, MIN_RESERVE_TOKENS, PRICE, validate


class FusionTests(unittest.TestCase):
    def test_identity_zero_mass_and_same_category_order(self):
        ids=np.array([[1,2,3,4]])
        logits=np.log([[.4,.3,.2,.1]])
        cats=[['a','a','b','b']]
        pred=[dict(categories=['b','a'],probabilities=[.9,.1])]
        np.testing.assert_array_equal(fuse(ids,logits,cats,pred,0),ids)
        rank=fuse(ids,logits,cats,pred,.5)[0].tolist()
        self.assertEqual([x for x in rank if x in (1,2)],[1,2])
        self.assertEqual([x for x in rank if x in (3,4)],[3,4])
        np.testing.assert_array_equal(fuse(ids,logits,cats,[dict(categories=['a','b'],probabilities=[0,0])],.5,2),ids)

    def test_log_formula_and_extreme_logits(self):
        ids=np.array([[1,2,3,4]])
        logits=np.log([[.4,.3,.2,.1]])
        q=np.array([.15,.85]); q=.98*q+.01; P=np.array([.7,.3])
        C=P**.65*q**(.35/2); C/=C.sum()
        expected=ids[:,np.argsort(-np.array([C[0]*4/7,C[0]*3/7,C[1]*2/3,C[1]/3]),kind='stable')]
        pred=[dict(categories=['a','b'],probabilities=[.15,.85])]
        np.testing.assert_array_equal(fuse(ids,logits,[['a','a','b','b']],pred,.35,2),expected)
        result=fuse(ids,np.array([[0,-1,-1000,-1001]]),[['a','a','b','b']],pred,.5,2)
        self.assertEqual(set(result[0]),set(ids[0]))

    def test_invalid_schema_fails(self):
        with self.assertRaises(ValueError):
            fuse([[1,2]],[[1,0]],[['a','b']],[dict(categories=['a','x'],probabilities=[.5,.5])],.5)
        with self.assertRaises(ValueError):
            fuse([[1]],[[1]],[['a']],[dict(categories=['a'],probabilities=[1])],.5,float('nan'))


class TransportTests(unittest.TestCase):
    def test_no_network_when_offline_or_budget_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload={'model':MODEL}
            with patch('requests.Session.post') as post:
                with self.assertRaises(FileNotFoundError): Client(tmp,1,True).query(payload)
                client=Client(tmp,MIN_RESERVE_TOKENS*PRICE)
                client.record('unknown',MIN_RESERVE_TOKENS*PRICE,status='reserved')
                with patch.dict('os.environ',{'TYPESAFE_API_KEY':'unit-test-only'}):
                    with self.assertRaises(RuntimeError): Client(tmp,MIN_RESERVE_TOKENS*PRICE).query(payload)
                post.assert_not_called()
            self.assertNotIn('unit-test-only',Path(tmp,'ledger.jsonl').read_text())

    def test_exact_option_matching(self):
        payload={'questions':{'next_category':{'criteria':{'k00':'a','outside_category':'b'}}}}
        response=dict(model=MODEL,answers={'next_category':dict(choice='k00',probabilities={'k00':.8,'outside_category':.2})})
        self.assertEqual(validate(payload,response),[.8,.2])
        response['answers']['next_category']['probabilities']['k00']=float('nan')
        with self.assertRaises(ValueError):validate(payload,response)


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        from rule_teacher import RankingBatch
        data=root/'data'; candidates=root/'candidates'; data.mkdir(); candidates.mkdir()
        events=[]
        for poi in range(20):
            events.append(dict(UserId=1,PoiId=poi,PoiCategoryName=f'category {poi}',SplitTag='train',
                UTCTimeOffsetEpoch=10,check_ins_id=poi,UTCTimeOffset='2010-01-01 00:00:10',last_checkin_epoch_time=0))
        pd.DataFrame(events).to_csv(data/'sample.csv',index=False)
        for split in ('validation','test'):
            queries=[dict(UserId=u,PoiId=1,PoiCategoryName='TARGET',UTCTimeOffsetEpoch=100,
                check_ins_id=1000+u,UTCTimeOffset='2010-01-01 00:01:40',last_checkin_epoch_time=10) for u in range(1,101)]
            name='validate_sample.csv' if split=='validation' else 'test_sample.csv'
            pd.DataFrame(queries).to_csv(data/name,index=False)
            RankingBatch(np.tile(np.arange(20),(100,1)),np.tile(-np.arange(20)*.01,(100,1)),
                np.ones(100,int),np.full(100,2),np.arange(100)).save(str(candidates/f'{split}_candidates.npz'))
        save(candidates/'candidate_manifest.json',dict(backbone='A4',dataset='nyc',smoke_only=False,
            candidate_hashes={s:sha256(candidates/f'{s}_candidates.npz') for s in ('validation','test')},
            sample_hash=sha256(data/'sample.csv'),query_hashes={s:sha256(data/n) for s,n in [('validation','validate_sample.csv'),('test','test_sample.csv')]}))
        return data,candidates

    def test_label_invariance_and_complete_offline_workflow(self):
        from argparse import Namespace
        from run_a5_jev import execute
        with tempfile.TemporaryDirectory() as tmp, patch('requests.Session.post') as network:
            root=Path(tmp); data,candidates=self.fixture(root); out=root/'output'
            args=Namespace(stage='prepare',dataset='nyc',data=data,candidates=candidates,output=out,cap_usd=.1,offline=True)
            with patch('builtins.print'): execute(args)
            args.stage='query-test'
            with self.assertRaises(FileNotFoundError): execute(args)
            d=Dataset('nyc',data,candidates,'validation'); before=d.request(0)
            d.queries.loc[0,'PoiId']=9999; d.queries.loc[0,'PoiCategoryName']='LEAK'
            self.assertEqual(before,d.request(0))
            future = d.history[1].iloc[:1].copy()
            future['UTCTimeOffsetEpoch'] = 1000
            future['CategoryText'] = 'FUTURE_LEAK'
            d.history[1] = pd.concat([d.history[1], future])
            self.assertEqual(before,d.request(0))
            # Synthetic cached responses test plumbing only; never report as Jev performance.
            for split in ('validation','test'):
                d=Dataset('nyc',data,candidates,split)
                for i in range(100):
                    payload,names=d.request(i); target=names.index('category 1')
                    probs={f'k{j:02d}':(.95 if j==target else .04/19) for j in range(20)}
                    probs['outside_category']=.01
                    response=dict(model=MODEL,answers={'next_category':dict(choice=f'k{target:02d}',probabilities=probs)})
                    save(out/'cache'/f'{digest(payload)}.json',dict(request_hash=digest(payload),request=payload,response=response))
            with patch('builtins.print'):
                for stage in ('query-validation','select','query-test','report'):
                    args.stage=stage; execute(args)
            self.assertTrue(json.loads((out/'frozen.json').read_text())['eligible_for_test'])
            self.assertEqual(json.loads((out/'test_report.json').read_text())['selected']['all']['net_top1'],100)
            with self.assertRaises(FileExistsError): execute(args)
            (data/'sample.csv').write_text((data/'sample.csv').read_text()+'\n')
            args.stage='query-validation'
            with self.assertRaises(ValueError):execute(args)
            network.assert_not_called()


if __name__=='__main__':unittest.main()
