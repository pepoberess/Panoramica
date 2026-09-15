"""Ítem 3.6: límites del canvas y homografías ajustadas, sin transformar imágenes."""

import numpy as np


def transform_corners(image, H):
    """Devuelve los cuatro bordes transformados como un array float64 (4, 2).

    Orden: (0, 0), (w, 0), (w, h), (0, h). Son límites geométricos,
    no índices de píxeles: x es columna/ancho, y es fila/alto.
    H transforma desde las coordenadas de esta imagen hacia el destino.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("La imagen debe ser un array de NumPy 2D o 3D, no None.")
    if image.size == 0:
        raise ValueError("La imagen no puede estar vacía.")

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("La homografía debe ser una matriz 3x3 con valores finitos.")

    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    # Mismo enfoque que homo/apply_transform/cart de los tutoriales.
    corners_homogeneous = np.hstack((corners, np.ones((4, 1))))
    transformed = (H @ corners_homogeneous.T).T
    denominator = transformed[:, 2]

    # Para volver a coordenadas cartesianas necesitamos dividir por w'.
    if np.any(denominator == 0):
        raise ValueError("No se puede dividir por una coordenada homogénea igual a cero.")

    corners_transformed = transformed[:, :2] / denominator[:, None]
    if not np.isfinite(corners_transformed).all():
        raise ValueError("Las esquinas transformadas deben tener coordenadas finitas.")
    return corners_transformed


def compute_panorama_bounds(
    img_left, img_anchor, img_right, H_left_to_anchor, H_right_to_anchor
):
    """Calcula el canvas rectangular para tres imágenes en el sistema del ancla.

    Retorna un diccionario con:
      size: (ancho, alto), enteros en el orden requerido por OpenCV.
      bounds: (xmin, ymin, xmax, ymax), redondeados hacia afuera.
      T, H_left_final, H_anchor_final, H_right_final: matrices 3x3.
      corners_anchor, corners_final: diccionarios con arrays (4, 2),
          con claves 'left', 'anchor', 'right', antes y después de T.

    Usamos límites (w, h), por eso tamaño = máximo - mínimo, sin sumar 1.
    El mínimo es respecto de límites enteros en el sistema del ancla.
    """
    images = {"left": img_left, "anchor": img_anchor, "right": img_right}
    homographies = {
        "left": np.asarray(H_left_to_anchor, dtype=np.float64),
        "anchor": np.eye(3),
        "right": np.asarray(H_right_to_anchor, dtype=np.float64),
    }
    corners_anchor = {
        name: transform_corners(image, homographies[name])
        for name, image in images.items()
    }
    all_corners = np.concatenate(list(corners_anchor.values()), axis=0)
    xmin, ymin = (int(value) for value in np.floor(all_corners.min(axis=0)))
    xmax, ymax = (int(value) for value in np.ceil(all_corners.max(axis=0)))
    panorama_width = xmax - xmin
    panorama_height = ymax - ymin
    if panorama_width <= 0 or panorama_height <= 0:
        raise ValueError("El ancho y el alto del canvas deben ser positivos.")

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    # Primero imagen -> ancla; después ancla -> canvas.
    final_homographies = {name: T @ H for name, H in homographies.items()}
    corners_final = {
        name: transform_corners(image, final_homographies[name])
        for name, image in images.items()
    }

    return {
        "size": (panorama_width, panorama_height),
        "bounds": (xmin, ymin, xmax, ymax),
        "T": T,
        "H_left_final": final_homographies["left"],
        "H_anchor_final": final_homographies["anchor"],
        "H_right_final": final_homographies["right"],
        "corners_anchor": corners_anchor,
        "corners_final": corners_final,
    }
