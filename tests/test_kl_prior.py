"""The Karhunen-Loeve basis the groundwater archive is generated against."""
from __future__ import annotations

import numpy as np

from lift.dataset.kl_prior import StuartKLPrior

GRID, K = 32, 10


def _prior():
    return StuartKLPrior(GRID, K=K, alpha=0.0, s=1.1, sigma=1.0)


def test_mode_count_is_k_squared():
    assert _prior().d == K * K


def test_amplitudes_are_positive_and_decay():
    amp = np.asarray(_prior().amp, float)
    assert amp.shape == (K * K,)
    assert (amp > 0).all()
    assert amp.max() == amp[0]


def test_zero_coefficients_give_the_zero_field():
    field = _prior().reconstruct(np.zeros(K * K))
    assert field.shape == (GRID, GRID)
    assert np.allclose(field, 0.0)


def test_field_is_linear_in_the_coefficients():
    kl = _prior()
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(K * K), rng.standard_normal(K * K)
    np.testing.assert_allclose(kl.reconstruct(a + 2.0 * b),
                               kl.reconstruct(a) + 2.0 * kl.reconstruct(b),
                               rtol=1e-10, atol=1e-10)
