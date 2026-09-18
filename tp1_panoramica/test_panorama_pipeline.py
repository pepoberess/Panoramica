"""Pruebas unitarias de geometría y blending del pipeline."""

import unittest

import cv2
import numpy as np

from tp1_panoramica.panorama.blending import (
    distance_weight,
    distance_weighted_blend,
    simple_average_blend,
    stitch_overwrite,
)
from tp1_panoramica.panorama.geometry import (
    compute_reprojection_error,
    dlt,
    ransac,
)


class GeometryTests(unittest.TestCase):
    """Verifica DLT normalizado, RANSAC y errores de reproyección."""

    def test_dlt_recovers_known_homography(self):
        """Recupera una homografía conocida con puntos sin ruido."""
        source = np.array(
            [[0, 0], [100, 0], [100, 80], [0, 80], [35, 20], [70, 55]],
            dtype=float,
        )
        expected = np.array(
            [[1.1, 0.08, 24], [-0.04, 0.95, 12], [0.0007, -0.0003, 1]],
            dtype=float,
        )
        homogeneous = np.column_stack([source, np.ones(len(source))])
        projected = (expected @ homogeneous.T).T
        destination = projected[:, :2] / projected[:, 2, None]
        estimated = dlt(source, destination)
        np.testing.assert_allclose(estimated, expected, atol=1e-9)
        np.testing.assert_allclose(
            compute_reprojection_error(estimated, source, destination),
            0,
            atol=1e-9,
        )

    def test_dlt_rejects_collinear_points(self):
        """Rechaza una configuración colineal degenerada."""
        source = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        with self.assertRaises(ValueError):
            dlt(source, source)

    def test_ransac_rejects_outliers(self):
        """Recupera el consenso y marca correspondencias alteradas."""
        generator = np.random.default_rng(4)
        source = generator.uniform(0, 500, size=(40, 2))
        expected = np.array(
            [[1.02, 0.03, 45], [-0.02, 0.98, -18], [0.0001, -0.0002, 1]],
            dtype=float,
        )
        homogeneous = np.column_stack([source, np.ones(len(source))])
        projected = (expected @ homogeneous.T).T
        destination = projected[:, :2] / projected[:, 2, None]
        destination += generator.normal(0, 0.15, size=destination.shape)
        destination[:8] = generator.uniform(-300, 800, size=(8, 2))
        estimated, mask = ransac(
            source, destination, num_iterations=1500, threshold=1.0, seed=7
        )
        errors = compute_reprojection_error(estimated, source, destination)
        np.testing.assert_array_equal(mask, errors < 1.0)
        self.assertGreaterEqual(mask.sum(), 30)
        self.assertLessEqual(mask[:8].sum(), 1)


class BlendingTests(unittest.TestCase):
    """Verifica validez, prioridad y normalización del blending."""

    def test_blends_keep_valid_black_pixels(self):
        """Distingue un píxel negro válido del exterior."""
        white = np.full((3, 3, 3), 200, dtype=np.uint8)
        black = np.zeros_like(white)
        mask = np.full((3, 3), 255, dtype=np.uint8)
        self.assertTrue(
            np.all(stitch_overwrite([white, black], [mask, mask]) == 0)
        )
        self.assertTrue(
            np.all(simple_average_blend([white, black], [mask, mask]) == 100)
        )
        weight = distance_weight(mask)
        blended, denominator = distance_weighted_blend(
            [white, black], [weight, weight]
        )
        self.assertTrue(np.all(blended == 100))
        self.assertTrue(np.all(denominator > 0))

    def test_masks_remain_binary_after_warping(self):
        """Comprueba la interpolación discreta de una máscara."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        transform = np.array(
            [[1, 0, 0.4], [0, 1, 0.4], [0, 0, 1]], dtype=float
        )
        warped = cv2.warpPerspective(mask, transform, (6, 6), flags=cv2.INTER_NEAREST)
        self.assertTrue(np.all(np.isin(np.unique(warped), [0, 255])))


if __name__ == "__main__":
    unittest.main()
