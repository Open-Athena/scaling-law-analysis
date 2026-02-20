"""Tests for experiment utilities."""

from scaling_law_analysis.experiments.exp5_parameter_recovery import _verify_default_eps


def test_lbfgsb_default_eps_matches_scipy():
    """LBFGSB_DEFAULT_EPS should match scipy's actual default."""
    _verify_default_eps()
