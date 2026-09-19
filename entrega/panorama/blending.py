"""Ítem 3.7: combinar imágenes ya deformadas al mismo canvas usando máscaras."""

import cv2
import numpy as np


def stitch_overwrite(images, masks):
    """Copia imágenes en orden y da prioridad a la última válida.

    Parámetros
    ----------
    images : list[np.ndarray]
        Imágenes del mismo canvas.
    masks : list[np.ndarray]
        Máscaras binarias asociadas.

    Retorna
    -------
    np.ndarray
        Panorama con pegado directo.
    """
    panorama = np.zeros_like(images[0])
    for image, mask in zip(images, masks):
        valid = mask > 0
        panorama[valid] = image[valid]
    return panorama


def simple_average_blend(images, masks):
    """Promedia las imágenes válidas y deja negro el exterior.

    Parámetros
    ----------
    images : list[np.ndarray]
        Imágenes del mismo canvas.
    masks : list[np.ndarray]
        Máscaras binarias asociadas.

    Retorna
    -------
    np.ndarray
        Panorama promediado en tipo ``uint8``.
    """
    accumulator = np.zeros(images[0].shape, dtype=np.float32)
    count = np.zeros(images[0].shape[:2], dtype=np.float32)
    for image, mask in zip(images, masks):
        valid = mask > 0
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
    """Calcula el peso según la distancia al exterior de una máscara.

    Parámetros
    ----------
    mask : np.ndarray
        Máscara con valores nulos fuera de la imagen.

    Retorna
    -------
    np.ndarray
        Pesos ``float32`` positivos dentro y nulos fuera.
    """
    binary = (mask > 0).astype(np.uint8)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1].copy()


def distance_weighted_blend(images, weights):
    """Combina imágenes con pesos y normaliza cada píxel.

    Parámetros
    ----------
    images : list[np.ndarray]
        Imágenes del mismo canvas.
    weights : list[np.ndarray]
        Pesos no negativos de cada imagen.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Panorama ``uint8`` y suma de pesos ``float32``.
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
