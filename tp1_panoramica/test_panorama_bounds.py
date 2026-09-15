"""Ejecutar desde la raíz: python3 -m unittest discover -s tp1_panoramica -v"""

from pathlib import Path
import unittest

import cv2
import numpy as np

from tp1_panoramica.panorama_bounds import compute_panorama_bounds, transform_corners


def translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


class PanoramaBoundsTests(unittest.TestCase):
    def test_identity_has_no_extra_row_or_column(self):
        img = np.zeros((40, 60, 3), dtype=np.uint8)
        result = compute_panorama_bounds(img, img, img, np.eye(3), np.eye(3))
        self.assertEqual(result["size"], (60, 40))
        np.testing.assert_array_equal(result["T"], np.eye(3))

    def test_translations_with_repository_image(self):
        path = Path(__file__).parent / "inputs" / "cuadro_1.jpg"
        img = cv2.imread(str(path))
        self.assertIsNotNone(img)
        h, w = img.shape[:2]
        result = compute_panorama_bounds(
            img, img, img, translation(-200, 0), translation(200, 0)
        )
        self.assertEqual(result["bounds"], (-200, 0, w + 200, h))
        self.assertEqual(result["size"], (w + 400, h))
        self.assertLess(result["corners_anchor"]["left"][:, 0].min(), 0)
        np.testing.assert_array_equal(result["T"], translation(200, 0))
        np.testing.assert_array_equal(result["H_anchor_final"], result["T"])
        corners = np.concatenate(list(result["corners_final"].values()))
        self.assertTrue(np.all(corners >= 0))
        self.assertTrue(np.all(corners <= np.array(result["size"])))
        np.testing.assert_allclose(corners.min(axis=0), [0, 0], atol=1e-7)
        np.testing.assert_allclose(corners.max(axis=0), result["size"], atol=1e-7)
        # Los extremos tocan los cuatro bordes: no se puede reducir el canvas.
        for name in ("left", "anchor", "right"):
            np.testing.assert_allclose(
                result["corners_final"][name],
                result["corners_anchor"][name] + [200, 0], atol=1e-7,
            )

    def test_fractional_limits_round_outward(self):
        img = np.zeros((10, 20), dtype=np.uint8)
        result = compute_panorama_bounds(
            img, img, img, translation(-2.25, -3.5), translation(5.2, 4.1)
        )
        self.assertEqual(result["bounds"], (-3, -4, 26, 15))
        self.assertEqual(result["size"], (29, 19))
        np.testing.assert_array_equal(result["T"], translation(3, 4))

    def test_projective_division(self):
        img = np.zeros((20, 40), dtype=np.uint8)
        H = np.array([[1, 0, -10], [0, 1, -5], [0.01, 0, 1]])
        expected = np.array([[-10, -5], [30 / 1.4, -5 / 1.4],
                             [30 / 1.4, 15 / 1.4], [-10, 15]])
        np.testing.assert_allclose(transform_corners(img, H), expected)

    def test_invalid_inputs(self):
        img = np.zeros((20, 40), dtype=np.uint8)
        for invalid_image in [None, np.zeros((0, 40))]:
            with self.subTest(image=invalid_image):
                with self.assertRaises(ValueError):
                    transform_corners(invalid_image, np.eye(3))
        for H in [np.eye(2), np.full((3, 3), np.nan), np.full((3, 3), np.inf),
                  # Denominador cero en las esquinas con x = 0.
                  np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])]:
            with self.subTest(H=H):
                with self.assertRaises(ValueError):
                    compute_panorama_bounds(img, img, img, H, np.eye(3))


if __name__ == "__main__":
    unittest.main()
