"""Carga, detección, descripción y asociación de características."""

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """Carga una imagen BGR y falla con un mensaje claro si no es legible.

    Parámetros
    ----------
    image_path : Path
        Ruta de la imagen.

    Retorna
    -------
    np.ndarray
        Imagen en el orden de canales BGR.
    """
    raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image


def load_images(image_paths: dict[int, Path]) -> dict[int, np.ndarray]:
    """Carga un conjunto de imágenes indexado y conserva sus claves.

    Parámetros
    ----------
    image_paths : dict[int, Path]
        Rutas asociadas a los índices de las imágenes.

    Retorna
    -------
    dict[int, np.ndarray]
        Imágenes BGR cargadas.
    """
    missing = [path for path in image_paths.values() if not path.is_file()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Faltan imágenes del dataset: {missing_names}")
    return {index: load_image(path) for index, path in image_paths.items()}


def load_triplet(input_dir: Path, dataset: str) -> dict[int, np.ndarray]:
    """Carga las imágenes ``<dataset>_0.jpg`` a ``<dataset>_2.jpg``.

    Parámetros
    ----------
    input_dir : Path
        Directorio que contiene las imágenes.
    dataset : str
        Prefijo común de los tres archivos.

    Retorna
    -------
    dict[int, np.ndarray]
        Tripleta BGR indexada de 0 a 2.
    """
    image_paths = {
        index: input_dir / f"{dataset}_{index}.jpg" for index in range(3)
    }
    return load_images(image_paths)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR a escala de grises.

    Parámetros
    ----------
    image : np.ndarray
        Imagen BGR o ya monocromática.

    Retorna
    -------
    np.ndarray
        Imagen de un canal.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_detector(name: str = "sift", nfeatures: int = 1500):
    """Crea un detector y descriptor SIFT u ORB.

    Parámetros
    ----------
    name : str
        Nombre del método.
    nfeatures : int
        Cantidad máxima de características solicitadas.

    Retorna
    -------
    cv2.Feature2D
        Detector configurado.
    """
    normalized_name = name.lower()
    if normalized_name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("Esta instalación de OpenCV no incluye SIFT.")
        return cv2.SIFT_create(nfeatures=nfeatures)
    if normalized_name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    raise ValueError("Detector no soportado. Usar 'sift' u 'orb'.")


def detect_and_describe(
    gray_image: np.ndarray, detector_name: str = "sift", nfeatures: int = 1500
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detecta puntos de interés y calcula sus descriptores.

    Parámetros
    ----------
    gray_image : np.ndarray
        Imagen en escala de grises.
    detector_name : str
        Nombre del detector y descriptor.
    nfeatures : int
        Cantidad máxima de puntos solicitados.

    Retorna
    -------
    tuple[list[cv2.KeyPoint], np.ndarray | None]
        Puntos detectados y descriptores asociados.
    """
    detector = create_detector(detector_name, nfeatures)
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def adaptive_non_maximal_suppression(
    keypoints: list[cv2.KeyPoint],
    descriptors: np.ndarray | None,
    num_features: int = 500,
    robustness: float = 0.9,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Selecciona puntos fuertes y distribuidos mediante A-NMS.

    Para cada punto se obtiene la distancia al punto más cercano que cumple
    ``respuesta_i < robustness * respuesta_j``. Luego se conservan los radios
    mayores y los descriptores se reordenan con los mismos índices.

    Parámetros
    ----------
    keypoints : list[cv2.KeyPoint]
        Puntos detectados.
    descriptors : np.ndarray | None
        Descriptores en el mismo orden.
    num_features : int
        Cantidad máxima de puntos a conservar.
    robustness : float
        Factor de robustez entre cero y uno.

    Retorna
    -------
    tuple[list[cv2.KeyPoint], np.ndarray | None]
        Puntos y descriptores seleccionados.
    """
    if not 0 < robustness <= 1:
        raise ValueError("robustness debe estar entre 0 (exclusivo) y 1.")
    if num_features < 1:
        raise ValueError("num_features debe ser mayor o igual a 1.")
    if descriptors is not None and len(descriptors) != len(keypoints):
        raise ValueError("La cantidad de descriptores no coincide con los keypoints.")
    if not keypoints:
        return [], None if descriptors is None else descriptors[:0]

    points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    responses = np.array(
        [keypoint.response for keypoint in keypoints], dtype=np.float32
    )
    radii = np.full(len(keypoints), np.inf, dtype=np.float32)

    for index, (point, response) in enumerate(zip(points, responses)):
        stronger = robustness * responses > response
        stronger[index] = False
        if np.any(stronger):
            distances = np.linalg.norm(points[stronger] - point, axis=1)
            radii[index] = distances.min()

    count = min(num_features, len(keypoints))
    selected_indices = np.argsort(-radii, kind="stable")[:count]
    selected_keypoints = [keypoints[index] for index in selected_indices]
    selected_descriptors = (
        None if descriptors is None else descriptors[selected_indices]
    )
    return selected_keypoints, selected_descriptors


def _norm_type_for_detector(detector_name: str) -> int:
    """Devuelve la distancia compatible con los descriptores elegidos."""
    normalized_name = detector_name.lower()
    if normalized_name == "sift":
        return cv2.NORM_L2
    if normalized_name == "orb":
        return cv2.NORM_HAMMING
    raise ValueError("Detector no soportado. Usar 'sift' u 'orb'.")


def match_with_lowe_ratio(
    query_descriptors: np.ndarray | None,
    train_descriptors: np.ndarray | None,
    detector_name: str = "sift",
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Asocia descriptores y conserva los que pasan Lowe ratio.

    Parámetros
    ----------
    query_descriptors : np.ndarray | None
        Descriptores de la imagen lateral.
    train_descriptors : np.ndarray | None
        Descriptores de la imagen ancla.
    detector_name : str
        Nombre del descriptor utilizado.
    ratio : float
        Umbral de Lowe entre cero y uno.

    Retorna
    -------
    list[cv2.DMatch]
        Correspondencias lateral hacia ancla.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio debe estar entre 0 y 1.")
    if query_descriptors is None or train_descriptors is None:
        return []
    if len(query_descriptors) == 0 or len(train_descriptors) < 2:
        return []

    matcher = cv2.BFMatcher(
        normType=_norm_type_for_detector(detector_name), crossCheck=False
    )
    nearest_neighbors = matcher.knnMatch(
        query_descriptors, train_descriptors, k=2
    )
    return [
        best_match
        for candidates in nearest_neighbors
        if len(candidates) == 2
        for best_match, second_match in [candidates]
        if best_match.distance < ratio * second_match.distance
    ]


def match_with_cross_check(
    query_descriptors: np.ndarray | None,
    train_descriptors: np.ndarray | None,
    detector_name: str = "sift",
) -> list[cv2.DMatch]:
    """Conserva asociaciones mutuas entre la imagen lateral y el ancla.

    Parámetros
    ----------
    query_descriptors : np.ndarray | None
        Descriptores de la imagen lateral.
    train_descriptors : np.ndarray | None
        Descriptores de la imagen ancla.
    detector_name : str
        Nombre del descriptor utilizado.

    Retorna
    -------
    list[cv2.DMatch]
        Correspondencias ordenadas por distancia.
    """
    if query_descriptors is None or train_descriptors is None:
        return []
    if len(query_descriptors) == 0 or len(train_descriptors) == 0:
        return []

    matcher = cv2.BFMatcher(
        normType=_norm_type_for_detector(detector_name), crossCheck=True
    )
    matches = matcher.match(query_descriptors, train_descriptors)
    return sorted(matches, key=lambda match: match.distance)


def match_with_lowe_and_cross_check(
    query_descriptors: np.ndarray | None,
    train_descriptors: np.ndarray | None,
    detector_name: str = "sift",
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Combina Lowe ratio con verificación cruzada.

    Parámetros
    ----------
    query_descriptors : np.ndarray | None
        Descriptores de la imagen lateral.
    train_descriptors : np.ndarray | None
        Descriptores de la imagen ancla.
    detector_name : str
        Nombre del descriptor utilizado.
    ratio : float
        Umbral de Lowe.

    Retorna
    -------
    list[cv2.DMatch]
        Intersección de ambas políticas.
    """
    lowe_matches = match_with_lowe_ratio(
        query_descriptors, train_descriptors, detector_name, ratio
    )
    cross_check_pairs = {
        (match.queryIdx, match.trainIdx)
        for match in match_with_cross_check(
            query_descriptors, train_descriptors, detector_name
        )
    }
    return [
        match
        for match in lowe_matches
        if (match.queryIdx, match.trainIdx) in cross_check_pairs
    ]


def match_descriptors(
    query_descriptors: np.ndarray | None,
    train_descriptors: np.ndarray | None,
    detector_name: str = "sift",
    ratio: float = 0.75,
    method: str = "lowe",
) -> list[cv2.DMatch]:
    """Asocia descriptores con Lowe, cross-check o ambos filtros.

    Parámetros
    ----------
    query_descriptors : np.ndarray | None
        Descriptores de la imagen lateral.
    train_descriptors : np.ndarray | None
        Descriptores de la imagen ancla.
    detector_name : str
        Nombre del descriptor utilizado.
    ratio : float
        Umbral de Lowe.
    method : str
        Política de filtrado.

    Retorna
    -------
    list[cv2.DMatch]
        Correspondencias filtradas.
    """
    normalized_method = method.lower()
    if normalized_method == "lowe":
        return match_with_lowe_ratio(
            query_descriptors, train_descriptors, detector_name, ratio
        )
    if normalized_method == "cross_check":
        return match_with_cross_check(
            query_descriptors, train_descriptors, detector_name
        )
    if normalized_method == "lowe_cross_check":
        return match_with_lowe_and_cross_check(
            query_descriptors, train_descriptors, detector_name, ratio
        )
    raise ValueError(
        "Método no soportado. Usar 'lowe', 'cross_check' o 'lowe_cross_check'."
    )


def match_triplet_to_anchor(
    descriptors: dict[int, np.ndarray | None],
    anchor_index: int = 1,
    detector_name: str = "sift",
    ratio: float = 0.75,
    method: str = "lowe",
) -> dict[int, list[cv2.DMatch]]:
    """Asocia cada imagen lateral con la imagen ancla.

    Parámetros
    ----------
    descriptors : dict[int, np.ndarray | None]
        Descriptores de todas las imágenes.
    anchor_index : int
        Índice de la imagen ancla.
    detector_name : str
        Nombre del descriptor utilizado.
    ratio : float
        Umbral de Lowe.
    method : str
        Política de filtrado.

    Retorna
    -------
    dict[int, list[cv2.DMatch]]
        Matches de cada lateral hacia el ancla.
    """
    if anchor_index not in descriptors:
        raise KeyError(f"No hay descriptores para la imagen ancla {anchor_index}.")
    return {
        index: match_descriptors(
            descriptors[index],
            descriptors[anchor_index],
            detector_name,
            ratio,
            method,
        )
        for index in descriptors
        if index != anchor_index
    }


def matched_point_coordinates(
    query_keypoints: list[cv2.KeyPoint],
    train_keypoints: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte matches en pares de coordenadas origen y ancla.

    Parámetros
    ----------
    query_keypoints : list[cv2.KeyPoint]
        Puntos de la imagen lateral.
    train_keypoints : list[cv2.KeyPoint]
        Puntos de la imagen ancla.
    matches : list[cv2.DMatch]
        Correspondencias entre ambas listas.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Coordenadas de origen y destino con forma ``(N, 2)``.
    """
    query_points = np.float32(
        [query_keypoints[match.queryIdx].pt for match in matches]
    ).reshape(-1, 2)
    train_points = np.float32(
        [train_keypoints[match.trainIdx].pt for match in matches]
    ).reshape(-1, 2)
    return query_points, train_points


def extract_triplet_features(
    images: dict[int, np.ndarray],
    detector_name: str = "sift",
    nfeatures: int = 1500,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, list[cv2.KeyPoint]],
    dict[int, np.ndarray | None],
]:
    """Convierte imágenes a gris y extrae sus características.

    Parámetros
    ----------
    images : dict[int, np.ndarray]
        Imágenes BGR indexadas.
    detector_name : str
        Nombre del detector y descriptor.
    nfeatures : int
        Cantidad máxima de puntos solicitados.

    Retorna
    -------
    tuple[dict, dict, dict]
        Imágenes grises, keypoints y descriptores.
    """
    grays = {index: to_grayscale(image) for index, image in images.items()}
    keypoints: dict[int, list[cv2.KeyPoint]] = {}
    descriptors: dict[int, np.ndarray | None] = {}
    for index, gray_image in grays.items():
        keypoints[index], descriptors[index] = detect_and_describe(
            gray_image, detector_name, nfeatures
        )
    return grays, keypoints, descriptors
