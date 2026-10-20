import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("cloud_ic", Path(__file__).resolve().parents[1] / "scripts/cloud_result_ic.py")
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)


class CloudInformationCriteriaTests(unittest.TestCase):
    def row(self):
        return dict(status="ok", model="llmsr", task="sr", n_train=20, n_test=10,
                    best_program="def equation(x, params):\n    unused = params[9]\n    a = params[2]\n    return a*x + params[2]\n",
                    method_provenance={"training_mse": 0.1}, test_mse=0.2,
                    expression_complexity=7, prediction_total_points=10,
                    prediction_finite_points=10, prediction_finite_coverage=1.0)

    def test_distinct_sparse_active_parameters(self):
        self.assertEqual(ic.parameter_count(self.row()["best_program"]), 1)
        self.assertEqual(ic.parameter_count("def f(x, params):\n    a,b=params[1:3]\n    a += b*x\n    return a\n"), 2)

    def test_training_sample_count_and_variance(self):
        row = self.row()
        result = ic.training_ic(row)
        self.assertEqual(result["information_criterion_k"], 2)
        self.assertEqual(result["information_criterion_n"], 20)
        row["n_train"] = 40
        self.assertNotEqual(ic.training_ic(row)["aic"], result["aic"])

    def test_zero_nonfinite_and_failed_abstain(self):
        for mse in (0, -1, math.inf, math.nan, True):
            row = self.row()
            row["method_provenance"]["training_mse"] = mse
            self.assertIsNone(ic.training_ic(row)["aic"])
        row = self.row()
        row["status"] = "error"
        self.assertIsNone(ic.supplement_cloud_row(row)["aic_proxy_heldout_complexity"])

    def test_proxy_does_not_fill_formal_columns_or_modify_source(self):
        row = self.row()
        row.pop("method_provenance")
        result = ic.supplement_cloud_row(row)
        self.assertIsNone(result["aic"])
        self.assertTrue(math.isfinite(result["aic_proxy_heldout_complexity"]))
        self.assertNotIn("aic", row)
        row["prediction_finite_coverage"] = 0.9
        self.assertIsNone(ic.proxy_ic(row)["aic_proxy_heldout_complexity"])

    def test_pde_domain_and_matching_points(self):
        row = self.row()
        row.update(task="pde", equation_residual_total_points=31,
                   equation_residual_points=31, equation_residual_finite_coverage=1,
                   equation_residual_mse=0.1)
        result = ic.proxy_ic(row)
        self.assertEqual(result["proxy_ic_n"], 31)
        self.assertEqual(result["proxy_ic_domain"], "heldout_equation_residual")
        row["equation_residual_points"] = 30
        self.assertIsNone(ic.proxy_ic(row)["aic_proxy_heldout_complexity"])


if __name__ == "__main__":
    unittest.main()
