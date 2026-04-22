import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    get_blurred_accuracy,
    get_mean_rating_error,
    evaluate_classification,
)


# ── get_blurred_accuracy ──────────────────────────────────────────────────────

class TestGetBlurredAccuracy:

    def test_exact_predictions(self):
        assert get_blurred_accuracy([1, 3, 7], [1, 3, 7]) == 1.0

    def test_one_step_off_counts_as_correct(self):
        assert get_blurred_accuracy([3], [4]) == 1.0
        assert get_blurred_accuracy([3], [2]) == 1.0

    def test_two_steps_off_counts_as_wrong(self):
        assert get_blurred_accuracy([3], [5]) == 0.0
        assert get_blurred_accuracy([3], [1]) == 0.0

    def test_mixed_predictions(self):
        # 2 exact + 1 off-by-one (all ≤ 1) + 1 off-by-two (wrong)
        result = get_blurred_accuracy([1, 2, 3, 4], [1, 3, 3, 7])
        assert result == pytest.approx(3 / 4)


# ── get_mean_rating_error ─────────────────────────────────────────────────────

class TestGetMeanRatingError:

    def test_perfect_predictions(self):
        assert get_mean_rating_error([1, 3, 7], [1, 3, 7]) == pytest.approx(0.0)

    def test_constant_error(self):
        assert get_mean_rating_error([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)

    def test_dataframe_input(self):
        y_df = pd.DataFrame({'OECD_RATING': [1, 3, 5]})
        assert get_mean_rating_error(y_df, [1, 3, 5]) == pytest.approx(0.0)

    def test_lower_is_better(self):
        good_pred = get_mean_rating_error([1, 7], [2, 6])
        bad_pred  = get_mean_rating_error([1, 7], [4, 4])
        assert good_pred < bad_pred


# ── evaluate_classification ───────────────────────────────────────────────────

class TestEvaluateClassification:

    def test_returns_expected_keys(self):
        metrics = evaluate_classification([1, 2, 3], [1, 2, 3], prefix='test_')
        expected_keys = {
            'test_accuracy', 'test_precision', 'test_recall',
            'test_f1', 'test_blurred_accuracy', 'test_mean_rating_error',
        }
        assert expected_keys == set(metrics.keys())

    def test_perfect_predictions_score_one(self):
        metrics = evaluate_classification([1, 2, 3, 4], [1, 2, 3, 4], prefix='')
        assert metrics['accuracy'] == pytest.approx(1.0)
        assert metrics['f1'] == pytest.approx(1.0)
        assert metrics['blurred_accuracy'] == pytest.approx(1.0)
        assert metrics['mean_rating_error'] == pytest.approx(0.0)

    def test_prefix_is_applied(self):
        metrics = evaluate_classification([1, 2], [1, 2], prefix='val_')
        assert all(k.startswith('val_') for k in metrics)