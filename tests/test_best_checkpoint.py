"""Unit tests for best-checkpoint metric helpers."""

from src.utils.best_checkpoint import (
    default_greater_is_better,
    is_valid_metric_value,
    metric_improves,
)


def test_default_greater_is_better_loss_like():
    assert default_greater_is_better("eval/loss") is False
    assert default_greater_is_better("eval/perplexity") is False
    assert default_greater_is_better("eval/l_ce") is False


def test_default_greater_is_better_accuracy_like():
    assert default_greater_is_better("eval/token_accuracy") is True
    assert default_greater_is_better("eval/f1") is True


def test_metric_improves():
    assert metric_improves(0.5, None, False) is True
    assert metric_improves(0.4, 0.5, False) is True
    assert metric_improves(0.6, 0.5, False) is False
    assert metric_improves(0.6, 0.5, True) is True


def test_is_valid_metric_value():
    assert is_valid_metric_value(1.0) is True
    assert is_valid_metric_value(float("nan")) is False
