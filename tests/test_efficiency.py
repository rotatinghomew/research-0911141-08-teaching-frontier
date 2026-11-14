import json
import unittest
from copy import deepcopy
from pathlib import Path

from teaching_frontier.domain import explore, validate
from teaching_frontier.efficiency import analyze_efficiency, parse_selection
from teaching_frontier.lp import solve_lp, LinearProgramError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "fixtures" / "data.json").read_text(encoding="utf-8"))
DEFAULT_SELECTION = {"family": "all", "constraint": "all", "quality_threshold": 0}


class LinearProgramTests(unittest.TestCase):
    def test_basic_minimization_with_inequalities(self):
        result = solve_lp([-1, -1], [([1, 1], "<=", 4), ([2, 1], "<=", 5)])
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["objective"], -4, places=7)
        self.assertAlmostEqual(result["x"][0], 1, places=7)
        self.assertAlmostEqual(result["x"][1], 3, places=7)

    def test_equality_and_greater_than_constraints(self):
        result = solve_lp([1, 1], [([1, 1], "=", 3), ([1, 0], ">=", 2)])
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["objective"], 3, places=7)

    def test_infeasible_program_is_reported(self):
        result = solve_lp([1], [([1], ">=", 2), ([1], "<=", 1)])
        self.assertEqual(result["status"], "infeasible")
        self.assertIsNone(result["objective"])

    def test_unbounded_program_is_reported(self):
        result = solve_lp([-1], [])
        self.assertEqual(result["status"], "unbounded")

    def test_negative_right_hand_side_is_normalized(self):
        result = solve_lp([1, 1], [([-1, -1], "=", -3), ([1, 0], ">=", 2)])
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["objective"], 3, places=7)

    def test_redundant_equality_row_does_not_break_phase_one(self):
        result = solve_lp([1], [([1], "=", 2), ([1], ">=", 1), ([1], "<=", 3)])
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["x"][0], 2, places=7)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(LinearProgramError):
            solve_lp([1, 2], [([1], "<=", 1)])


class EfficiencyStructureTests(unittest.TestCase):
    def setUp(self):
        self.view = analyze_efficiency(FIXTURE, DEFAULT_SELECTION)

    def test_payload_is_json_serializable_without_nan(self):
        json.dumps(self.view, allow_nan=False, ensure_ascii=False)

    def test_explore_includes_efficiency_and_keeps_ratio_section(self):
        view = explore(FIXTURE, DEFAULT_SELECTION)
        self.assertIn("efficiency", view)
        self.assertIn("rows", view)
        self.assertTrue(view["rows"])

    def test_analysis_does_not_mutate_dataset(self):
        before = deepcopy(FIXTURE)
        analyze_efficiency(FIXTURE, DEFAULT_SELECTION)
        self.assertEqual(before, FIXTURE)

    def test_scores_stay_within_unit_interval(self):
        for row in self.view["results"]:
            if row["status"] == "evaluated":
                for key in ("score_crs", "score_vrs", "scale_efficiency", "score_adjusted_vrs"):
                    self.assertIsNotNone(row[key])
                    self.assertGreaterEqual(row[key], 0.0)
                    self.assertLessEqual(row[key], 1.0 + 1e-9)

    def test_crs_is_never_looser_than_vrs(self):
        for row in self.view["results"]:
            if row["status"] == "evaluated":
                self.assertLessEqual(row["score_crs"], row["score_vrs"] + 1e-9)
                self.assertLessEqual(row["score_adjusted_vrs"], row["score_vrs"] + 1e-9)

    def test_scale_efficiency_equals_crs_over_vrs(self):
        for row in self.view["results"]:
            if row["status"] == "evaluated":
                self.assertAlmostEqual(row["scale_efficiency"],
                                       row["score_crs"] / row["score_vrs"], places=6)

    def test_peer_weights_are_non_negative_and_sum_to_one_under_vrs(self):
        for row in self.view["results"]:
            if row["status"] != "evaluated":
                continue
            self.assertTrue(row["peers"])
            for peer in row["peers"]:
                self.assertGreaterEqual(peer["weight"], 0.0)
            self.assertAlmostEqual(sum(peer["weight"] for peer in row["peers"]), 1.0, places=5)

    def test_input_gaps_explain_every_active_input_with_units(self):
        method = self.view["method"]
        for row in self.view["results"]:
            if row["status"] != "evaluated":
                continue
            gap_fields = {gap["field"] for gap in row["input_gaps"]}
            self.assertEqual(set(method["active_inputs"]) | set(method["nondiscretionary"]), gap_fields)
            for gap in row["input_gaps"]:
                self.assertIn("unit", gap)
                if gap["discretionary"]:
                    self.assertIsNotNone(gap["target_vrs"])
                    self.assertGreaterEqual(gap["slack"], -1e-9)

    def test_method_distinguishes_discretionary_and_fixed_inputs(self):
        self.assertEqual(self.view["method"]["nondiscretionary"], ["space"])
        self.assertIn("staff_hours", self.view["method"]["active_inputs"])

    def test_every_gap_and_output_target_carries_its_raw_unit(self):
        for row in self.view["results"]:
            if row["status"] != "evaluated":
                continue
            for gap in row["input_gaps"]:
                self.assertEqual(gap["unit"], self.view["method"]["units"].get(gap["field"], ""))
            for target in row["output_targets"]:
                self.assertEqual(target["unit"], self.view["method"]["units"].get(target["field"], ""))

    def test_reference_access_flag_present_and_explained(self):
        for row in self.view["results"]:
            if row["status"] != "evaluated":
                self.assertIsNone(row["reference_access"])
                continue
            access = row["reference_access"]
            self.assertIn("reachable", access)
            self.assertTrue(access["message"])
            flagged_labels = {gap["label"] for gap in row["input_gaps"]
                              if gap.get("reference_unreachable")}
            self.assertEqual(set(access["unreachable_fields"]), flagged_labels)

    def test_high_gate_surfaces_unreachable_reference_without_presenting_cut_target(self):
        view = analyze_efficiency(FIXTURE, {**DEFAULT_SELECTION, "quality_threshold": 0.85})
        unreachable = [row for row in view["results"]
                       if row.get("reference_access") and not row["reference_access"]["reachable"]]
        self.assertTrue(unreachable)
        for row in unreachable:
            self.assertTrue(any("高于" in text for text in view["boundaries"]))
            above = [gap for gap in row["input_gaps"] if gap.get("reference_unreachable")]
            self.assertTrue(above)
            for gap in above:
                self.assertGreater(gap["target_vrs"], gap["current"])
                self.assertLess(gap["gap_ratio"], 0)


class BoundaryTests(unittest.TestCase):
    def test_missing_funding_excludes_only_funding_without_imputation(self):
        view = analyze_efficiency(FIXTURE, DEFAULT_SELECTION)
        self.assertIn("funding", view["method"]["excluded_inputs"])
        self.assertNotIn("funding", view["method"]["active_inputs"])
        self.assertTrue(any("经费" in text for text in view["boundaries"]))

    def test_zero_output_center_is_kept_but_not_a_reference(self):
        view = analyze_efficiency(FIXTURE, DEFAULT_SELECTION)
        zero = next(row for row in view["results"] if row["center_id"] == "center_00")
        self.assertFalse(zero["reference_eligible"])
        self.assertIn("真实零", zero["reference_reason"])
        # 自身仍接受评估，零值保留在原始数据与输出目标中
        self.assertEqual(zero["status"], "evaluated")
        completed = next(target for target in zero["output_targets"] if target["field"] == "completed")
        self.assertEqual(completed["current"], 0.0)

    def test_missing_output_marks_center_non_reference(self):
        data = deepcopy(FIXTURE)
        next(center for center in data["centers"] if center["center_id"] == "center_03")["completed"] = None
        view = analyze_efficiency(data, DEFAULT_SELECTION)
        row = next(item for item in view["results"] if item["center_id"] == "center_03")
        self.assertFalse(row["reference_eligible"])
        self.assertIn("产出缺报", row["reference_reason"])

    def test_quality_gate_removes_references_but_keeps_evaluation(self):
        view = analyze_efficiency(FIXTURE, {**DEFAULT_SELECTION, "quality_threshold": 0.9})
        weak = next(row for row in view["results"] if row["center_id"] == "center_13")
        self.assertFalse(weak["reference_eligible"])
        self.assertEqual(weak["status"], "evaluated")
        for peer in weak["peers"]:
            reference = next(center for center in FIXTURE["centers"] if center["center_id"] == peer["center_id"])
            self.assertGreaterEqual(reference["quality"], 0.9 - 1e-9)

    def test_over_restricted_comparison_reports_no_comparable_objects(self):
        view = analyze_efficiency(FIXTURE, {"family": "design", "constraint": "remote",
                                           "quality_threshold": 0.95})
        self.assertEqual(view["summary"]["可参照对象"], 0)
        self.assertTrue(all(row["status"] == "no_comparable" for row in view["results"]))
        for row in view["results"]:
            self.assertTrue(row["status_reason"])
        # 汇总层面明确列出哪些中心缺少可比对象及原因
        joined = " ".join(view["boundaries"])
        self.assertIn("缺少可比对象的中心", joined)
        for row in view["results"]:
            self.assertIn(row["center_id"], joined)

    def test_small_reference_set_carries_discrimination_warning(self):
        view = analyze_efficiency(FIXTURE, {"family": "design", "constraint": "remote",
                                           "quality_threshold": 0.8})
        self.assertTrue(any("区分度" in text for text in view["boundaries"]))


class StabilityTests(unittest.TestCase):
    def test_threshold_zero_has_no_stability_change_payload(self):
        view = analyze_efficiency(FIXTURE, DEFAULT_SELECTION)
        self.assertEqual(view["comparison"]["quality_threshold"], 0.0)
        self.assertTrue(all(row["stability"] is None for row in view["results"]))

    def test_higher_threshold_produces_baseline_comparison(self):
        view = analyze_efficiency(FIXTURE, {**DEFAULT_SELECTION, "quality_threshold": 0.9})
        moved = [row for row in view["results"] if row["stability"] is not None]
        self.assertTrue(moved)
        for row in moved:
            stability = row["stability"]
            self.assertEqual(stability["baseline_quality_threshold"], 0.0)
            self.assertIsInstance(stability["peers_kept"], list)
            self.assertIsInstance(stability["stable"], bool)

    def test_invalid_selection_values_fall_back_to_defaults(self):
        parsed = parse_selection({"family": "unknown", "constraint": "ocean",
                                  "quality_threshold": 9}, FIXTURE)
        self.assertEqual(parsed, {"family": "all", "constraint": "all", "quality_threshold": 1.0})
        parsed = parse_selection({"quality_threshold": -3}, FIXTURE)
        self.assertEqual(parsed["quality_threshold"], 0.0)


class UnitInvarianceTests(unittest.TestCase):
    def _complete_funding(self, data):
        data = deepcopy(data)
        for center in data["centers"]:
            if center["funding"] is None:
                center["funding"] = 38800
        return data

    def test_scores_peers_and_weights_unchanged_under_rescaling(self):
        complete = self._complete_funding(FIXTURE)
        selection = {**DEFAULT_SELECTION, "quality_threshold": 0.85}
        base = analyze_efficiency(complete, selection)
        rescaled_data = self._complete_funding(FIXTURE)
        for center in rescaled_data["centers"]:
            center["funding"] /= 10000.0
            center["staff_hours"] *= 100
            center["equipment_hours"] *= 100
            center["space"] /= 10
        rescaled = analyze_efficiency(rescaled_data, selection)
        base_rows = {row["center_id"]: row for row in base["results"]}
        scaled_rows = {row["center_id"]: row for row in rescaled["results"]}
        for center_id, row in base_rows.items():
            other = scaled_rows[center_id]
            for key in ("score_crs", "score_vrs", "scale_efficiency", "score_adjusted_vrs"):
                if row[key] is None:
                    self.assertIsNone(other[key])
                else:
                    self.assertAlmostEqual(row[key], other[key], places=9)
            self.assertEqual([peer["center_id"] for peer in row["peers"]],
                             [peer["center_id"] for peer in other["peers"]])
            for peer, scaled_peer in zip(row["peers"], other["peers"]):
                self.assertAlmostEqual(peer["weight"], scaled_peer["weight"], places=9)

    def test_raw_unit_targets_move_with_the_unit_change(self):
        complete = self._complete_funding(FIXTURE)
        base = analyze_efficiency(complete, DEFAULT_SELECTION)
        rescaled_data = self._complete_funding(FIXTURE)
        for center in rescaled_data["centers"]:
            center["funding"] /= 10000.0
        rescaled = analyze_efficiency(rescaled_data, DEFAULT_SELECTION)
        base_rows = {row["center_id"]: row for row in base["results"]}
        scaled_rows = {row["center_id"]: row for row in rescaled["results"]}
        for center_id, row in base_rows.items():
            if row["status"] != "evaluated":
                continue
            gap = next(item for item in row["input_gaps"] if item["field"] == "funding")
            scaled_gap = next(item for item in scaled_rows[center_id]["input_gaps"]
                              if item["field"] == "funding")
            self.assertAlmostEqual(gap["current"] / 10000.0, scaled_gap["current"], places=4)
            self.assertAlmostEqual(gap["target_vrs"] / 10000.0, scaled_gap["target_vrs"], places=4)


class DeterminismTests(unittest.TestCase):
    def test_repeated_analysis_returns_identical_payload(self):
        first = json.dumps(analyze_efficiency(FIXTURE, DEFAULT_SELECTION),
                           ensure_ascii=False, sort_keys=True)
        second = json.dumps(analyze_efficiency(deepcopy(FIXTURE), DEFAULT_SELECTION),
                            ensure_ascii=False, sort_keys=True)
        self.assertEqual(first, second)

    def test_fixture_remains_valid(self):
        validate(FIXTURE)


if __name__ == "__main__":
    unittest.main()
