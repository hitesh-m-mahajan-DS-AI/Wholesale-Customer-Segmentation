"""Checks the feature boundary, reproducibility, profiles and silhouette arithmetic."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import pairwise_distances

from main import FEATURES, fit_segments, load_data, profile_segments, validate_spending


class SegmentationChecks(unittest.TestCase):
    def test_source_integrity(self):
        data, audit = load_data()
        self.assertEqual(len(data), 440)
        self.assertEqual(audit["missing_values"], 0)
        self.assertEqual(audit["exact_duplicate_rows"], 0)
        self.assertTrue(data["customer_id"].is_unique)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.csv"
            path.write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(path)

    def test_invalid_spending_is_rejected(self):
        data, _ = load_data()
        for invalid in [-1, np.nan, np.inf]:
            changed = data.copy()
            changed["Fresh"] = changed["Fresh"].astype(float)
            changed.loc[0, "Fresh"] = invalid
            with self.assertRaises(ValueError):
                validate_spending(changed)

    def test_reproducible_full_assignment_and_preprocessing(self):
        data, _ = load_data()
        first, assigned, transformed, score = fit_segments(data)
        _, second, _, second_score = fit_segments(data)
        np.testing.assert_array_equal(assigned["cluster"], second["cluster"])
        self.assertAlmostEqual(score, second_score)
        np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)
        self.assertEqual(first.named_steps["kmeans"].n_features_in_, 6)
        self.assertEqual(set(assigned["cluster"]), {1, 2, 3})
        self.assertEqual(len(assigned), len(data))

    def test_metadata_does_not_influence_clusters(self):
        data, _ = load_data()
        _, original, _, _ = fit_segments(data)
        changed = data.copy()
        changed["Channel"] = 3 - changed["Channel"]
        changed["Region"] = 999
        changed["total_category_spending"] = 0
        _, other, _, _ = fit_segments(changed)
        np.testing.assert_array_equal(original["cluster"], other["cluster"])

    def test_profiles_reconcile_and_use_original_units(self):
        data, _ = load_data()
        _, customers, _, _ = fit_segments(data)
        summary, medians, _, ratios = profile_segments(customers)
        self.assertEqual(summary["customers"].sum(), 440)
        self.assertAlmostEqual(summary["share_pct"].sum(), 100)
        for cluster in summary.index:
            group = customers[customers["cluster"] == cluster]
            np.testing.assert_allclose(medians.loc[cluster], group[FEATURES].median())
            np.testing.assert_allclose(ratios.loc[cluster], medians.loc[cluster] / data[FEATURES].median())

    def test_silhouette_matches_independent_distance_formula(self):
        data, _ = load_data()
        _, customers, transformed, score = fit_segments(data)
        distances = pairwise_distances(transformed)
        labels = customers["cluster"].to_numpy()
        calculated = []
        for i in range(len(labels)):
            own = np.flatnonzero(labels == labels[i])
            own = own[own != i]
            a = distances[i, own].mean()
            b = min(distances[i, labels == label].mean() for label in np.unique(labels) if label != labels[i])
            calculated.append((b - a) / max(a, b))
        np.testing.assert_allclose(calculated, customers["silhouette"], atol=1e-10)
        self.assertAlmostEqual(np.mean(calculated), score)


if __name__ == "__main__":
    unittest.main()
