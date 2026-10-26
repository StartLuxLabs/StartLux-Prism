"""Timing normalization must not hide user-named questions or option probabilities."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from amd_compare_service import semantic

class SemanticTests(unittest.TestCase):
    def test_only_schema_timing_is_ignored(self):
        before={'latency_ms':{'type':'choice','choice':'ms','details':{
            'first_pass':{'probabilities':{'ms':.75,'seconds':.25},'ms':20},
            'thought':{'text':'answer','tokens':2,'ms':30},'total_ms':50}}}
        after=copy.deepcopy(before)
        after['latency_ms']['details']['first_pass']['ms']=40
        after['latency_ms']['details']['total_ms']=70
        self.assertEqual(semantic(before),semantic(after))
        after['latency_ms']['details']['first_pass']['probabilities']['ms']=.8
        self.assertNotEqual(semantic(before),semantic(after))
        self.assertIn('latency_ms',semantic(before))
        self.assertEqual(before['latency_ms']['details']['total_ms'],50)

if __name__=='__main__':unittest.main()
