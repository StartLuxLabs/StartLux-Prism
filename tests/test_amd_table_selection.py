"""Selection must not depend on stage order or silently transfer old qualification."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from amd_table_selection import select_row, merge_qualified


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


if __name__ == '__main__':unittest.main()
