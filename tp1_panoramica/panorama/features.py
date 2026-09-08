"""Carga de imágenes y extracción inicial de features para el TP1."""

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """Carga una imagen BGR y falla con un mensaje claro si no es legible."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image


def load_triplet(input_dir: Path, dataset: str) -> dict[int, np.ndarray]:
    """Carga las tres imágenes ``<dataset>_0.jpg`` a ``<dataset>_2.jpg``.

    La imagen 1 se utilizará como ancla en las etapas posteriores.
    """
    image_paths = {
        index: input_dir / f"{dataset}_{index}.jpg" for index in range(3)
    }
    missing = [path for path in image_paths.values() if not path.is_file()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Faltan imágenes del dataset: {missing_names}")

    return {index: load_image(path) for index, path in image_paths.items()}


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR de OpenCV a escala de grises."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_detector(name: str = "sift", nfeatures: int = 1500):
    """Crea el detector y descriptor elegido para las primeras pruebas.

    SIFT es la opción por defecto porque suele ser robusta para este dataset y
    genera descriptores flotantes comparables con distancia L2.
    """
    # 
    normalized_name = name.lower()
    if normalized_name == "sift":
        if not hasattr(cv2, "SIFT_create"): # SIFT: histogramas de gradientes
            raise RuntimeError("Esta instalación de OpenCV no incluye SIFT.")
        return cv2.SIFT_create(nfeatures=nfeatures)
    if normalized_name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    raise ValueError("Detector no soportado. Usar 'sift' u 'orb'.")


def detect_and_describe(
    gray_image: np.ndarray, detector_name: str = "sift", nfeatures: int = 1500
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detecta keypoints y calcula sus descriptores en una imagen en gris."""
    detector = create_detector(detector_name, nfeatures)
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def adaptive_non_maximal_suppression(
    keypoints: list[cv2.KeyPoint],
    descriptors: np.ndarray | None,
    num_features: int = 500,
    robustness: float = 0.9,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Selecciona keypoints fuertes y espacialmente distribuidos mediante A-NMS.

    Para cada punto se calcula el radio hasta el keypoint más cercano cuya
    respuesta sea suficientemente mayor (``response_j > robustness * response_i``).
    Se conservan los ``num_features`` puntos con mayor radio adaptativo. Los
    descriptores devueltos mantienen el mismo orden que los keypoints elegidos.
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
    responses = np.array([keypoint.response for keypoint in keypoints], dtype=np.float32)
    radii = np.full(len(keypoints), np.inf, dtype=np.float32)

    for index, (point, response) in enumerate(zip(points, responses)):
        stronger = responses > robustness * response
        stronger[index] = False  # Un keypoint no puede suprimir a sí mismo.
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


def extract_triplet_features(
    images: dict[int, np.ndarray], detector_name: str = "sift", nfeatures: int = 1500
) -> tuple[dict[int, np.ndarray], dict[int, list[cv2.KeyPoint]], dict[int, np.ndarray | None]]:
    """Convierte las tres imágenes a gris y extrae features de cada una."""
    grays = {index: to_grayscale(image) for index, image in images.items()}
    keypoints: dict[int, list[cv2.KeyPoint]] = {}
    descriptors: dict[int, np.ndarray | None] = {}

    for index, gray_image in grays.items():
        keypoints[index], descriptors[index] = detect_and_describe(
            gray_image, detector_name, nfeatures
        )

    return grays, keypoints, descriptors
