"""Ítem 3.7.1: warping individual de imágenes y máscaras al canvas común."""

import cv2
import numpy as np


def warp_image_to_canvas(image, H, size, interpolation):
    """Aplica H final a una imagen; size usa el orden (ancho, alto).

    H debe incluir la traslación calculada en 3.6. La interpolación se recibe
    explícitamente para poder comparar métodos. Fuera de la imagen se usa 0.
    Acepta imágenes de color o máscaras 2D; no combina imágenes.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("La imagen debe ser un array 2D o 3D no vacío.")
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("H debe ser una matriz 3x3 con valores finitos.")
    if len(size) != 2 or any(
        not isinstance(value, (int, np.integer)) or value <= 0 for value in size
    ):
        raise ValueError("size debe contener dos enteros positivos: (ancho, alto).")

    return cv2.warpPerspective(
        image, H, (int(size[0]), int(size[1])), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def warp_valid_mask(image, H, size):
    """Transforma una máscara de validez: 255 dentro de la imagen, 0 fuera.

    Se usa NEAREST para conservar valores binarios. La validez depende del
    soporte de la imagen original, no de si sus píxeles son negros.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("La imagen debe ser un array 2D o 3D no vacío.")
    h, w = image.shape[:2]
    mask_original = np.ones((h, w), dtype=np.uint8) * 255
    return warp_image_to_canvas(mask_original, H, size, cv2.INTER_NEAREST)
