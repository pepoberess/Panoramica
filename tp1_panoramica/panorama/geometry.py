"""Estimación de homografías y geometría del canvas panorámico."""

import numpy as np


def _validate_correspondences(spts, dpts, minimum=4):
    """Valida y convierte dos conjuntos de correspondencias."""
    source = np.asarray(spts, dtype=np.float64)
    destination = np.asarray(dpts, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,):
        raise ValueError("Los puntos de origen deben tener forma (N, 2).")
    if destination.shape != source.shape:
        raise ValueError("Origen y destino deben tener la misma forma.")
    if len(source) < minimum:
        raise ValueError(f"Se necesitan al menos {minimum} correspondencias.")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("Las correspondencias deben ser finitas.")
    return source, destination


def _normalization_transform(points):
    """Normaliza puntos a centro cero y distancia media raíz de dos."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_distance = np.linalg.norm(centered, axis=1).mean()
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("Los puntos no permiten estimar una homografía.")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def dlt(spts, dpts):
    """Estima una homografía con DLT normalizado, sin usar OpenCV.

    Parámetros
    ----------
    spts : np.ndarray
        Puntos de origen con forma ``(N, 2)`` y ``N >= 4``.
    dpts : np.ndarray
        Puntos de destino correspondientes.

    Retorna
    -------
    np.ndarray
        Homografía ``H`` de 3 por 3 tal que ``destino ~ H @ origen``.
    """
    source, destination = _validate_correspondences(spts, dpts)
    if np.linalg.matrix_rank(np.column_stack([source, np.ones(len(source))])) < 3:
        raise ValueError("Los puntos de origen no pueden ser colineales.")
    if np.linalg.matrix_rank(
        np.column_stack([destination, np.ones(len(destination))])
    ) < 3:
        raise ValueError("Los puntos de destino no pueden ser colineales.")

    source_normalized, source_transform = _normalization_transform(source)
    destination_normalized, destination_transform = _normalization_transform(
        destination
    )
    rows = []
    for (x_coord, y_coord), (x_dest, y_dest) in zip(
        source_normalized, destination_normalized
    ):
        rows.append(
            [
                -x_coord,
                -y_coord,
                -1.0,
                0.0,
                0.0,
                0.0,
                x_dest * x_coord,
                x_dest * y_coord,
                x_dest,
            ]
        )
        rows.append(
            [
                0.0,
                0.0,
                0.0,
                -x_coord,
                -y_coord,
                -1.0,
                y_dest * x_coord,
                y_dest * y_coord,
                y_dest,
            ]
        )
    matrix = np.asarray(rows, dtype=np.float64)
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    normalized_homography = right_vectors[-1].reshape(3, 3)
    homography = (
        np.linalg.inv(destination_transform)
        @ normalized_homography
        @ source_transform
    )
    divisor = homography[2, 2]
    if abs(divisor) <= np.finfo(np.float64).eps:
        divisor = np.linalg.norm(homography)
    if abs(divisor) <= np.finfo(np.float64).eps:
        raise ValueError("DLT produjo una homografía degenerada.")
    homography = homography / divisor
    if not np.isfinite(homography).all():
        raise ValueError("DLT produjo una homografía no finita.")
    return homography


def compute_reprojection_error(H, spts, dpts):
    """Calcula el error euclídeo de reproyección para cada par.

    Parámetros
    ----------
    H : np.ndarray
        Homografía de origen hacia destino.
    spts : np.ndarray
        Puntos de origen con forma ``(N, 2)``.
    dpts : np.ndarray
        Puntos de destino correspondientes.

    Retorna
    -------
    np.ndarray
        Error por correspondencia, medido en píxeles.
    """
    source, destination = _validate_correspondences(spts, dpts, minimum=1)
    homography = np.asarray(H, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("H debe ser una matriz 3x3 con valores finitos.")
    homogeneous = np.column_stack([source, np.ones(len(source))])
    projected_homogeneous = (homography @ homogeneous.T).T
    denominators = projected_homogeneous[:, 2]
    errors = np.full(len(source), np.inf, dtype=np.float64)
    valid = np.abs(denominators) > np.finfo(np.float64).eps
    projected = projected_homogeneous[valid, :2] / denominators[valid, None]
    errors[valid] = np.linalg.norm(projected - destination[valid], axis=1)
    return errors


def ransac(spts, dpts, num_iterations=2000, threshold=5.0, seed=None):
    """Estima una homografía robusta con RANSAC implementado manualmente.

    Parámetros
    ----------
    spts : np.ndarray
        Correspondencias de origen potencialmente ruidosas.
    dpts : np.ndarray
        Correspondencias de destino.
    num_iterations : int
        Cantidad de muestras aleatorias de cuatro pares.
    threshold : float
        Error máximo en píxeles para considerar un inlier.
    seed : int | None
        Semilla opcional para reproducibilidad.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Homografía refinada y máscara booleana de inliers.
    """
    source, destination = _validate_correspondences(spts, dpts)
    if num_iterations < 1:
        raise ValueError("num_iterations debe ser positivo.")
    if threshold <= 0:
        raise ValueError("threshold debe ser positivo.")

    random_generator = np.random.default_rng(seed)
    best_mask = None
    best_count = 0
    best_mean_error = np.inf

    for _ in range(num_iterations):
        sample_indices = random_generator.choice(
            len(source), size=4, replace=False
        )
        try:
            candidate = dlt(source[sample_indices], destination[sample_indices])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = compute_reprojection_error(candidate, source, destination)
        candidate_mask = errors < threshold
        candidate_count = int(candidate_mask.sum())
        candidate_mean = (
            float(errors[candidate_mask].mean())
            if candidate_count >= 4
            else np.inf
        )
        if candidate_count > best_count or (
            candidate_count == best_count and candidate_mean < best_mean_error
        ):
            best_mask = candidate_mask
            best_count = candidate_count
            best_mean_error = candidate_mean

    if best_mask is None or best_count < 4:
        raise RuntimeError("RANSAC no encontró un modelo con cuatro inliers.")

    current_mask = best_mask
    homography = dlt(source[current_mask], destination[current_mask])
    for _ in range(50):
        new_mask = compute_reprojection_error(
            homography, source, destination
        ) < threshold
        if new_mask.sum() < 4:
            break
        if np.array_equal(new_mask, current_mask):
            current_mask = new_mask
            break
        current_mask = new_mask
        homography = dlt(source[current_mask], destination[current_mask])
    homography = dlt(source[current_mask], destination[current_mask])
    final_mask = compute_reprojection_error(
        homography, source, destination
    ) < threshold
    if final_mask.sum() >= 4 and not np.array_equal(final_mask, current_mask):
        homography = dlt(source[final_mask], destination[final_mask])
        current_mask = compute_reprojection_error(
            homography, source, destination
        ) < threshold
    else:
        current_mask = final_mask
    return homography, current_mask


def transform_corners(image, H):
    """Transforma los cuatro límites geométricos de una imagen.

    Parámetros
    ----------
    image : np.ndarray
        Imagen cuya forma define ancho y alto.
    H : np.ndarray
        Homografía hacia el sistema de destino.

    Retorna
    -------
    np.ndarray
        Esquinas transformadas con forma ``(4, 2)``.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("La imagen debe ser un array 2D o 3D.")
    if image.size == 0:
        raise ValueError("La imagen no puede estar vacía.")
    homography = np.asarray(H, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("La homografía debe ser una matriz 3x3 finita.")

    height, width = image.shape[:2]
    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([corners, np.ones(4)])
    transformed = (homography @ homogeneous.T).T
    denominator = transformed[:, 2]
    if np.any(np.abs(denominator) <= np.finfo(np.float64).eps):
        raise ValueError("Una esquina se proyecta al infinito.")
    cartesian = transformed[:, :2] / denominator[:, None]
    if not np.isfinite(cartesian).all():
        raise ValueError("Las esquinas transformadas deben ser finitas.")
    return cartesian


def compute_panorama_bounds(
    img_left, img_anchor, img_right, H_left_to_anchor, H_right_to_anchor
):
    """Calcula el canvas entero mínimo y ajusta las homografías.

    Parámetros
    ----------
    img_left, img_anchor, img_right : np.ndarray
        Imágenes izquierda, central y derecha.
    H_left_to_anchor, H_right_to_anchor : np.ndarray
        Homografías de cada lateral hacia el ancla.

    Retorna
    -------
    dict
        Tamaño, límites, traslación, homografías y esquinas.
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
    minimum_x, minimum_y = (
        int(value) for value in np.floor(all_corners.min(axis=0))
    )
    maximum_x, maximum_y = (
        int(value) for value in np.ceil(all_corners.max(axis=0))
    )
    panorama_width = maximum_x - minimum_x
    panorama_height = maximum_y - minimum_y
    if panorama_width <= 0 or panorama_height <= 0:
        raise ValueError("El ancho y el alto del canvas deben ser positivos.")

    translation = np.array(
        [[1, 0, -minimum_x], [0, 1, -minimum_y], [0, 0, 1]],
        dtype=np.float64,
    )
    final_homographies = {
        name: translation @ homography
        for name, homography in homographies.items()
    }
    corners_final = {
        name: transform_corners(image, final_homographies[name])
        for name, image in images.items()
    }
    return {
        "size": (panorama_width, panorama_height),
        "bounds": (minimum_x, minimum_y, maximum_x, maximum_y),
        "T": translation,
        "H_left_final": final_homographies["left"],
        "H_anchor_final": final_homographies["anchor"],
        "H_right_final": final_homographies["right"],
        "corners_anchor": corners_anchor,
        "corners_final": corners_final,
    }
