import argparse

import numpy as np
import scipy.ndimage as ndi

import test_utils as tutils
from eaa.image_proc import error_minimization_registration


class TestErrorMinimizationRegistration(tutils.BaseTester):

    def test_zero_shift(self):
        """Identical images should return a shift close to zero.

        The local quadratic fit can be offset from zero by up to ~0.25 pixels
        on random images due to natural asymmetry of the MSE surface around the
        minimum.  The rounded result is always the correct integer (0, 0).
        """
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((64, 64))
        shift = error_minimization_registration(ref, ref, y_valid_fraction=0.5, x_valid_fraction=0.5)
        assert np.allclose(shift, [0.0, 0.0], atol=0.3)

    def test_integer_shifts(self):
        """Integer shifts are recovered to within ±0.3 pixels by the local quadratic fit.

        The local MSE neighbourhood is slightly asymmetric for random images, which
        biases the quadratic minimum by up to ~0.25 pixels away from the true integer.
        Rounding the returned value always gives the correct integer.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((64, 64))
        cases = [(2, -3), (-1, 4), (3, 3), (-4, -2), (0, 5)]
        for true_dy, true_dx in cases:
            moving = np.roll(np.roll(ref, true_dy, axis=0), true_dx, axis=1)
            shift = error_minimization_registration(
                moving, ref, y_valid_fraction=0.5, x_valid_fraction=0.5
            )
            assert np.allclose(shift, [-true_dy, -true_dx], atol=0.3), (
                f"true=({true_dy},{true_dx})  got shift={shift}"
            )

    def test_roll_recovers_ref(self):
        """roll(moving, shift) must equal ref for integer shifts."""
        rng = np.random.default_rng(7)
        ref = rng.standard_normal((64, 64))
        for true_dy, true_dx in [(2, -3), (-1, 4), (4, -4)]:
            moving = np.roll(np.roll(ref, true_dy, axis=0), true_dx, axis=1)
            shift = error_minimization_registration(
                moving, ref, y_valid_fraction=0.5, x_valid_fraction=0.5
            )
            recovered = np.roll(
                np.roll(moving, int(round(shift[0])), axis=0),
                int(round(shift[1])), axis=1,
            )
            assert np.allclose(recovered, ref, atol=1e-10), (
                f"roll(moving, shift) != ref for true=({true_dy},{true_dx})"
            )

    def test_subpixel_shift(self):
        """Quadratic refinement should recover sub-pixel shifts.

        A sharp Gaussian bump is used (sigma=3) so the MSE landscape has high
        curvature near the minimum. A tight valid fraction (0.9) keeps the search
        range small (±3 pixels), so the quadratic fit is dominated by points near
        the minimum rather than the flat tails of the error surface.
        map_coordinates with mode='nearest' avoids wrap-around boundary artefacts.
        """
        h, w = 64, 64
        y, x = np.ogrid[:h, :w]
        ref = np.exp(-((y - h // 2) ** 2 + (x - w // 2) ** 2) / (2.0 * 3.0 ** 2))
        true_dy, true_dx = 0.4, -0.7
        yg, xg = np.meshgrid(
            np.arange(h, dtype=float) - true_dy,
            np.arange(w, dtype=float) - true_dx,
            indexing="ij",
        )
        moving = ndi.map_coordinates(ref, [yg, xg], order=1, mode="nearest")
        shift = error_minimization_registration(
            moving, ref, y_valid_fraction=0.9, x_valid_fraction=0.9
        )
        assert np.allclose(shift, [-true_dy, -true_dx], atol=0.05), (
            f"sub-pixel: expected ({-true_dy},{-true_dx}), got {shift}"
        )

    def test_no_margin_returns_zeros(self):
        """valid_fraction=1.0 leaves no margin; the function should return (0, 0)."""
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((64, 64))
        moving = np.roll(ref, 3, axis=0)
        shift = error_minimization_registration(
            moving, ref, y_valid_fraction=1.0, x_valid_fraction=1.0
        )
        assert np.allclose(shift, [0.0, 0.0])

    def test_asymmetric_valid_fractions(self):
        """Asymmetric valid fractions restrict the search range on each axis independently."""
        rng = np.random.default_rng(9)
        ref = rng.standard_normal((64, 64))
        true_dy, true_dx = 2, 0
        moving = np.roll(ref, true_dy, axis=0)
        # x_valid_fraction=1.0 → no x margin, only y shift can be detected
        shift = error_minimization_registration(
            moving, ref, y_valid_fraction=0.5, x_valid_fraction=1.0
        )
        assert np.isclose(shift[0], -true_dy, atol=1e-2)
        assert np.isclose(shift[1], 0.0, atol=1e-2)

    def test_noise_robustness(self):
        """Moderate additive noise should not prevent integer-precision recovery."""
        rng = np.random.default_rng(5)
        ref = rng.standard_normal((64, 64))
        true_dy, true_dx = 3, -2
        moving = np.roll(np.roll(ref, true_dy, axis=0), true_dx, axis=1)
        moving = moving + rng.standard_normal(moving.shape) * 0.3
        shift = error_minimization_registration(
            moving, ref, y_valid_fraction=0.5, x_valid_fraction=0.5
        )
        assert np.allclose(shift, [-true_dy, -true_dx], atol=0.5), (
            f"noisy: expected ({-true_dy},{-true_dx}), got {shift}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    tester = TestErrorMinimizationRegistration()
    tester.setup_method(name="", debug=True)
    tester.test_zero_shift()
    tester.test_integer_shifts()
    tester.test_roll_recovers_ref()
    tester.test_subpixel_shift()
    tester.test_no_margin_returns_zeros()
    tester.test_asymmetric_valid_fractions()
    tester.test_noise_robustness()
    print("All tests passed.")
