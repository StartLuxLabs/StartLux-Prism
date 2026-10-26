"""The server's thought limit applies to omitted and explicit request budgets."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startlux_prism.think import AutoThink, default_controls, parse_controls


class ThinkBudgetTests(unittest.TestCase):
    def setUp(self):
        self.defaults = default_controls(AutoThink(budget=4096))

    def test_omitted_budget_respects_server_limit(self):
        for body in ({}, {"think_budget": None}, {"think_mode": "always"}):
            with self.subTest(body=body):
                self.assertEqual(parse_controls(body, self.defaults, 256).budget, 256)
        self.assertEqual(self.defaults.budget, 4096)

    def test_explicit_budget_keeps_existing_clamping(self):
        for requested, expected in ((17, 17), (256, 256), (4096, 256)):
            with self.subTest(requested=requested):
                self.assertEqual(parse_controls({"think_budget": requested}, self.defaults, 256).budget, expected)

    def test_lower_folder_default_is_preserved(self):
        defaults = default_controls(AutoThink(budget=128))
        self.assertEqual(parse_controls({}, defaults, 256).budget, 128)

    def test_unrestricted_default_and_other_controls_are_preserved(self):
        self.assertEqual(parse_controls({}, self.defaults, 4096), self.defaults)
        controls = parse_controls({"think_mode": "never", "gate": .2, "mix": .4}, self.defaults, 256)
        self.assertEqual((controls.budget, controls.think_mode, controls.gate, controls.mix), (256, "never", .2, .4))

    def test_invalid_explicit_budget_still_rejected(self):
        for value in (0, -1, True, 1.5, "256"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_controls({"think_budget": value}, self.defaults, 256)


if __name__ == "__main__":
    unittest.main()
