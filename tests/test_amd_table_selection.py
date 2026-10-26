"""Selection must not depend on stage order or silently transfer old qualification."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from amd_table_selection import select_row, select_fixtures, merge_qualified


def candidate(cold, hot, partial, correct=True):
    return dict(cold_ms=cold, hot_ms=hot, partial_ms=partial, correct=correct,
                implementation='dot', cfg=[4, 8, 4])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(key='q4k:8:4096:4096:0:4096', family='decode', baseline='old',
            candidates={'old': candidate(10,10,10), 'dot': candidate(5,5,5),
                        'last_wmma': candidate(8,8,8)})

    def test_best_qualified_family_wins_in_any_order(self):
        self.assertEqual(select_row(self.row)['winner'], 'dot')
        self.row['candidates'] = dict(reversed(list(self.row['candidates'].items())))
        self.assertEqual(select_row(self.row)['winner'], 'dot')

    def test_bad_partial_or_numerical_results_are_not_promoted(self):
        self.row['candidates']['dot']['partial_ms'] = 11
        self.row['candidates']['last_wmma']['correct'] = False
        self.assertTrue(select_row(self.row)['retained_baseline'])

    def test_missing_partial_evidence_is_not_silently_accepted(self):
        del self.row['candidates']['dot']['partial_ms']
        with self.assertRaises(KeyError): select_row(self.row)

    def test_source_changes_require_requalification(self):
        base=dict(kernel_identity='old',runtime={},entries={})
        data=dict(kernel_identity='new',runtime={},protocol='real-weight-seven-paired-v1',
                  completed=True,experiment='test',rows=[self.row])
        with self.assertRaises(ValueError):merge_qualified(data,base,'new',{})

    def test_existing_table_is_preserved_without_inplace_mutation(self):
        base=dict(kernel_identity='same',runtime={},entries={'legacy-key':{'implementation':'wmma'}})
        original=copy.deepcopy(base)
        data=dict(kernel_identity='same',runtime={},protocol='real-weight-seven-paired-v1',
                  completed=True,experiment='test',rows=[self.row])
        result,decisions=merge_qualified(data,base,'same',{})
        self.assertEqual(base,original)
        self.assertEqual(result['entries']['legacy-key'],original['entries']['legacy-key'])
        self.assertEqual(decisions[0]['winner'],'dot')

    def test_w4a16_retains_fallback_without_an_invalid_table_entry(self):
        row=copy.deepcopy(self.row)
        for name in ('dot','last_wmma'): row['candidates'][name]['correct']=False
        data=dict(profile='w4a16',kernel_identity='same',runtime={},protocol='real-weight-seven-paired-v1',
                  completed=True,experiment='test',rows=[row])
        base=dict(kernel_identity='same',runtime={},entries={})
        result,decisions=merge_qualified(data,base,'same',{})
        self.assertEqual(result['entries'],{})
        self.assertTrue(decisions[0]['retained_baseline'])

    def test_w4a16_promotes_only_valid_metadata_choices(self):
        row=copy.deepcopy(self.row)
        data=dict(profile='w4a16',kernel_identity='same',runtime={},protocol='real-weight-seven-paired-v1',
                  completed=True,experiment='test',rows=[row])
        base=dict(kernel_identity='same',runtime={},entries={})
        with self.assertRaises(ValueError):merge_qualified(data,base,'same',{})
        row['candidates']['dot'].update(implementation='meta',cfg=[16,64,4,1,32])
        result,_=merge_qualified(data,base,'same',{})
        self.assertEqual(result['entries'][row['key']],[16,64,4,1,32])

    def test_shared_fixture_winner_must_pass_every_input(self):
        second=copy.deepcopy(self.row)
        second['candidates']['dot']['partial_ms']=11
        for rows in ([self.row,second],[second,self.row]):
            result=select_fixtures(rows)
            self.assertEqual(result['winner'],'last_wmma')
            self.assertEqual(result['fixture_count'],2)
        second['candidates']['last_wmma']['correct']=False
        self.assertTrue(select_fixtures([self.row,second])['retained_baseline'])

    def test_fixture_ranking_normalizes_each_runs_baseline(self):
        second=copy.deepcopy(self.row)
        for value in second['candidates'].values():
            for key in ('cold_ms','hot_ms','partial_ms'):value[key]*=100
        self.assertEqual(select_fixtures([self.row,second])['winner'],'dot')
        second['candidates']['old']['cfg']=[8,8,4]
        with self.assertRaises(ValueError):select_fixtures([self.row,second])

    def test_quarantine_blocks_conflicting_promotion(self):
        data=dict(protocol='real-weight-seven-paired-v1', completed=True,
                  kernel_identity='same', runtime={}, rows=[self.row],
                  quarantined=[dict(key=self.row['key'])])
        base=dict(kernel_identity='same',runtime={},entries={})
        with self.assertRaises(ValueError):merge_qualified(data,base,'same',{})
        data['rows']=[];base['entries'][self.row['key']]={}
        with self.assertRaises(ValueError):merge_qualified(data,base,'same',{})
        base['entries']={}
        table,decisions=merge_qualified(data,base,'same',{})
        self.assertEqual(table['entries'],{})
        self.assertEqual(decisions[0]['status'],'quarantined')


if __name__ == '__main__':unittest.main()
