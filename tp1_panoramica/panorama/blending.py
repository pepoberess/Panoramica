"""Ítem 3.7: combinar imágenes ya deformadas al mismo canvas usando máscaras."""

import cv2
import numpy as np


def stitch_overwrite(images, masks):
    """Copia en el orden recibido; la última imagen válida tiene prioridad.

    images: imágenes uint8 de igual shape (alto, ancho, 3).
    masks: máscaras del mismo canvas, 0 fuera y 255 dentro de cada imagen.
    Los píxeles negros también se copian si su máscara es válida.
    """
    panorama = np.zeros_like(images[0])
    for image, mask in zip(images, masks):
        valid = mask > 0
        panorama[valid] = image[valid]
    return panorama


def simple_average_blend(images, masks):
    """Promedia sólo las imágenes válidas en cada píxel; exterior negro."""
    accumulator = np.zeros(images[0].shape, dtype=np.float32)
    count = np.zeros(images[0].shape[:2], dtype=np.float32)
    for image, mask in zip(images, masks):
        valid = mask > 0
        # El destino float32 evita sumar intensidades en uint8.
        np.add(accumulator, image, out=accumulator, where=valid[..., None])
        count += valid
    assert np.isfinite(accumulator).all() and np.isfinite(count).all()
    np.divide(accumulator, count[..., None], out=accumulator,
              where=count[..., None] > 0)
    assert np.isfinite(accumulator).all()
    np.rint(accumulator, out=accumulator)
    np.clip(accumulator, 0, 255, out=accumulator)
    return accumulator.astype(np.uint8)


def distance_weight(mask):
    """Peso float32 = distancia al exterior de una máscara, cero fuera.

    Un borde de ceros de un píxel hace que el límite del canvas también cuente
    como exterior, incluso cuando la máscara válida llega hasta ese límite.
    DIST_L2 con máscara 5 aproxima la distancia euclídea.
    """
    binary = (mask > 0).astype(np.uint8)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1].copy()


def distance_weighted_blend(images, weights):
    """Retorna (panorama uint8, suma de pesos float32).

    Los pesos provienen de distance_weight: positivos dentro de cada máscara
    y cero fuera. Se expanden sobre los tres canales y se normalizan por píxel.
    Donde no hay peso, el panorama queda negro.
    """
    accumulator = np.zeros(images[0].shape, dtype=np.float32)
    denominator = np.zeros(images[0].shape[:2], dtype=np.float32)
    for image, weight in zip(images, weights):
        assert np.isfinite(weight).all() and np.all(weight >= 0)
        accumulator += np.multiply(image, weight[..., None], dtype=np.float32)
        denominator += weight
    assert np.isfinite(accumulator).all() and np.isfinite(denominator).all()
    np.divide(accumulator, denominator[..., None], out=accumulator,
              where=denominator[..., None] > 0)
    assert np.isfinite(accumulator).all()
    np.rint(accumulator, out=accumulator)
    np.clip(accumulator, 0, 255, out=accumulator)
    return accumulator.astype(np.uint8), denominator
