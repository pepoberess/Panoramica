"""Orquestación reutilizable de los experimentos del trabajo práctico."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .blending import (
    distance_weight,
    distance_weighted_blend,
    simple_average_blend,
    stitch_overwrite,
)
from .features import (
    adaptive_non_maximal_suppression,
    extract_triplet_features,
    load_image,
    load_images,
    load_triplet,
    match_triplet_to_anchor,
    matched_point_coordinates,
)
from .geometry import (
    compute_panorama_bounds,
    compute_reprojection_error,
    dlt,
    ransac,
)
from .warping import warp_image_to_canvas, warp_valid_mask


MATCHING_LABELS = {
    "lowe": "Lowe ratio",
    "cross_check": "Cross-check",
    "lowe_cross_check": "Lowe ratio + cross-check",
}


@dataclass(frozen=True)
class ConfiguracionTP:
    """Agrupa los parámetros reproducibles del pipeline."""

    project_dir: Path
    input_dir: Path
    output_dir: Path
    anchor_index: int = 1
    detector: str = "sift"
    nfeatures: int = 1500
    anms_features: int = 500
    anms_robustness: float = 0.9
    lowe_ratio: float = 0.75
    matching_method: str = "lowe_cross_check"
    ransac_iterations: int = 2000
    ransac_threshold: float = 5.0
    ransac_seed: int = 0
    interpolation: int = cv2.INTER_LINEAR


@dataclass
class AnalisisCaracteristicas:
    """Conserva resultados de features, A-NMS y matching."""

    nombre: str
    images: dict
    grays: dict
    keypoints: dict
    descriptors: dict
    anms_keypoints: dict
    anms_descriptors: dict
    matching_experiments: dict
    selected_method: str
    matches: dict
    matched_points: dict
    anchor_index: int


@dataclass
class ResultadoGeometria:
    """Conserva homografías, máscaras y errores de RANSAC."""

    homographies: dict
    inlier_masks: dict
    errors: dict
    threshold: float


@dataclass
class ResultadoPanorama:
    """Conserva las etapas de warping y blending de una secuencia."""

    analysis: AnalisisCaracteristicas
    geometry: ResultadoGeometria
    bounds: dict
    warps: dict
    masks: dict
    weights: dict
    blends: dict
    selected_method: str
    final: np.ndarray


def crear_configuracion(current_dir=None):
    """Localiza el proyecto y crea la configuración central del TP.

    Parámetros
    ----------
    current_dir : Path | None
        Directorio desde el cual se ejecuta el notebook.

    Retorna
    -------
    ConfiguracionTP
        Rutas y parámetros utilizados por todos los experimentos.
    """
    starting_point = Path.cwd() if current_dir is None else Path(current_dir)
    candidates = [starting_point, starting_point / "entrega"]
    project_dir = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "panorama").is_dir()
            and (candidate / "imgs").is_dir()
        ),
        None,
    )
    if project_dir is None:
        raise FileNotFoundError(
            "No se encontró la raíz del proyecto (debe contener panorama/ e imgs/)."
        )
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return ConfiguracionTP(
        project_dir=project_dir,
        input_dir=project_dir / "imgs",
        output_dir=output_dir,
    )


def _prepare_analysis(nombre, images, config):
    """Ejecuta features, A-NMS y los tres métodos de matching."""
    grays, keypoints, descriptors = extract_triplet_features(
        images,
        detector_name=config.detector,
        nfeatures=config.nfeatures,
    )
    anms_keypoints = {}
    anms_descriptors = {}
    for index in sorted(images):
        anms_keypoints[index], anms_descriptors[index] = (
            adaptive_non_maximal_suppression(
                keypoints[index],
                descriptors[index],
                num_features=config.anms_features,
                robustness=config.anms_robustness,
            )
        )
    matching_experiments = {
        method: match_triplet_to_anchor(
            anms_descriptors,
            anchor_index=config.anchor_index,
            detector_name=config.detector,
            ratio=config.lowe_ratio,
            method=method,
        )
        for method in MATCHING_LABELS
    }
    matches = matching_experiments[config.matching_method]
    matched_points = {}
    for source_index, source_matches in matches.items():
        source_points, anchor_points = matched_point_coordinates(
            anms_keypoints[source_index],
            anms_keypoints[config.anchor_index],
            source_matches,
        )
        matched_points[source_index] = {
            "source": source_points,
            "anchor": anchor_points,
        }
    return AnalisisCaracteristicas(
        nombre=nombre,
        images=images,
        grays=grays,
        keypoints=keypoints,
        descriptors=descriptors,
        anms_keypoints=anms_keypoints,
        anms_descriptors=anms_descriptors,
        matching_experiments=matching_experiments,
        selected_method=config.matching_method,
        matches=matches,
        matched_points=matched_points,
        anchor_index=config.anchor_index,
    )


def cargar_y_analizar_dataset(nombre, config):
    """Carga una tripleta provista y ejecuta features y matching.

    Parámetros
    ----------
    nombre : str
        Prefijo de los archivos JPEG.
    config : ConfiguracionTP
        Configuración del experimento.

    Retorna
    -------
    AnalisisCaracteristicas
        Resultados hasta las coordenadas asociadas.
    """
    images = load_triplet(config.input_dir, nombre)
    return _prepare_analysis(nombre, images, config)


def cargar_y_analizar_propias(config):
    """Carga la secuencia propia y ejecuta features y matching.

    Parámetros
    ----------
    config : ConfiguracionTP
        Configuración del experimento.

    Retorna
    -------
    AnalisisCaracteristicas
        Resultados de la secuencia propia.
    """
    image_paths = {
        0: config.input_dir / "imagenpropia_izq.jpeg",
        1: config.input_dir / "imagenpropia_ancla.jpeg",
        2: config.input_dir / "imagenpropia_der.jpeg",
    }
    return _prepare_analysis("imágenes propias", load_images(image_paths), config)


def evaluar_metodos_matching(analysis, config):
    """Evalúa cada política de matching con el mismo RANSAC.

    Parámetros
    ----------
    analysis : AnalisisCaracteristicas
        Features y matches del dataset.
    config : ConfiguracionTP
        Parámetros geométricos comunes.

    Retorna
    -------
    list[dict]
        Cantidades, proporción de inliers y error por método y par.
    """
    rows = []
    for method, matches_by_source in analysis.matching_experiments.items():
        for source_index, matches in matches_by_source.items():
            source_points, anchor_points = matched_point_coordinates(
                analysis.anms_keypoints[source_index],
                analysis.anms_keypoints[analysis.anchor_index],
                matches,
            )
            if len(matches) < 4:
                rows.append(
                    {
                        "method": method,
                        "source_index": source_index,
                        "matches": len(matches),
                        "inliers": 0,
                        "percentage": 0.0,
                        "mean_error": np.nan,
                    }
                )
                continue
            homography, mask = ransac(
                source_points,
                anchor_points,
                num_iterations=config.ransac_iterations,
                threshold=config.ransac_threshold,
                seed=config.ransac_seed,
            )
            errors = compute_reprojection_error(
                homography, source_points, anchor_points
            )
            rows.append(
                {
                    "method": method,
                    "source_index": source_index,
                    "matches": len(matches),
                    "inliers": int(mask.sum()),
                    "percentage": 100.0 * float(mask.mean()),
                    "mean_error": float(errors[mask].mean()),
                }
            )
    return rows


def correspondencias_manuales_cuadro():
    """Devuelve cuatro pares manuales para cada lateral del dataset Cuadro.

    Retorna
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Coordenadas lateral y ancla para los pares 0 hacia 1 y 2 hacia 1.
    """
    return {
        0: (
            np.float64(
                [[2979, 1803], [2166, 2619], [2494, 1888], [2494, 1497]]
            ),
            np.float64(
                [[2147, 2508], [589, 2866], [1324, 2301], [1498, 1853]]
            ),
        ),
        2: (
            np.float64(
                [[940, 2186], [568, 1619], [1460, 1546], [407, 1421]]
            ),
            np.float64(
                [[2243, 2467], [1602, 1909], [2966, 1690], [1283, 1695]]
            ),
        ),
    }

def correspondencias_manuales_udesa():
    """Devuelve cuatro pares manuales para cada lateral del dataset Cuadro.

    Retorna
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Coordenadas lateral y ancla para los pares 0 hacia 1 y 2 hacia 1.
    """
    return {
        0: (
            np.float64(
                [[2295, 1161], [1595, 1557], [2383, 1628], [2599, 1497]]
            ),
            np.float64(
                [[1411, 1222], [717, 1620], [1509, 1682], [1721, 1552]]
            ),
        ),
        2: (
            np.float64(
                [[601, 1724], [711, 1900], [1394, 1726], [1069, 1495]]
            ),
            np.float64(
                [[2015, 1739], [2133, 1911], [2811, 1721], [2465, 1500]]
            ),
        ),
    }


def estimar_homografias_manuales(correspondences):
    """Calcula DLT y error para los pares elegidos manualmente.

    Parámetros
    ----------
    correspondences : dict
        Pares de coordenadas por imagen lateral.

    Retorna
    -------
    dict[int, dict]
        Homografía y errores de reproyección de cada par.
    """
    results = {}
    for source_index, (source_points, anchor_points) in correspondences.items():
        homography = dlt(source_points, anchor_points)
        results[source_index] = {
            "H": homography,
            "errors": compute_reprojection_error(
                homography, source_points, anchor_points
            ),
        }
    return results


def comparar_thresholds(analysis, config, thresholds=(3.0, 5.0, 8.0)):
    """Compara umbrales de RANSAC sobre las mismas correspondencias.

    Parámetros
    ----------
    analysis : AnalisisCaracteristicas
        Dataset con los matches seleccionados.
    config : ConfiguracionTP
        Iteraciones y semilla comunes.
    thresholds : tuple[float, ...]
        Umbrales de reproyección a ensayar.

    Retorna
    -------
    list[dict]
        Estadísticas por par y umbral.
    """
    rows = []
    for source_index, points in analysis.matched_points.items():
        source = points["source"]
        anchor = points["anchor"]
        for threshold in thresholds:
            homography, mask = ransac(
                source,
                anchor,
                num_iterations=config.ransac_iterations,
                threshold=threshold,
                seed=config.ransac_seed,
            )
            errors = compute_reprojection_error(homography, source, anchor)
            rows.append(
                {
                    "source_index": source_index,
                    "threshold": float(threshold),
                    "total": len(source),
                    "inliers": int(mask.sum()),
                    "percentage": 100.0 * float(mask.mean()),
                    "mean_error": float(errors[mask].mean()),
                    "median_error": float(np.median(errors[mask])),
                }
            )
    return rows


def estimar_geometria(analysis, config, threshold=None):
    """Estima las dos homografías finales mediante RANSAC propio.

    Parámetros
    ----------
    analysis : AnalisisCaracteristicas
        Dataset con correspondencias seleccionadas.
    config : ConfiguracionTP
        Iteraciones, umbral y semilla.
    threshold : float | None
        Umbral alternativo opcional.

    Retorna
    -------
    ResultadoGeometria
        Homografías, inliers y errores de todos los matches.
    """
    selected_threshold = (
        config.ransac_threshold if threshold is None else float(threshold)
    )
    homographies = {}
    masks = {}
    errors = {}
    for source_index, points in analysis.matched_points.items():
        homography, mask = ransac(
            points["source"],
            points["anchor"],
            num_iterations=config.ransac_iterations,
            threshold=selected_threshold,
            seed=config.ransac_seed,
        )
        homographies[source_index] = homography
        masks[source_index] = mask
        errors[source_index] = compute_reprojection_error(
            homography, points["source"], points["anchor"]
        )
    return ResultadoGeometria(
        homographies=homographies,
        inlier_masks=masks,
        errors=errors,
        threshold=selected_threshold,
    )


def verificar_limites_controlados():
    """Comprueba identidad, traslaciones y redondeo del canvas.

    Retorna
    -------
    dict
        Resultados de identidad, traslación y límites fraccionales.
    """
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    identity = compute_panorama_bounds(
        image, image, image, np.eye(3), np.eye(3)
    )
    left = np.array([[1, 0, -20], [0, 1, 0], [0, 0, 1]], dtype=float)
    right = np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], dtype=float)
    translations = compute_panorama_bounds(
        image, image, image, left, right
    )
    small = np.zeros((10, 20), dtype=np.uint8)
    fractional_left = np.array(
        [[1, 0, -2.25], [0, 1, -3.5], [0, 0, 1]], dtype=float
    )
    fractional_right = np.array(
        [[1, 0, 5.2], [0, 1, 4.1], [0, 0, 1]], dtype=float
    )
    fractional = compute_panorama_bounds(
        small, small, small, fractional_left, fractional_right
    )
    assert identity["size"] == (60, 40)
    assert translations["bounds"] == (-20, 0, 80, 40)
    assert translations["size"] == (100, 40)
    assert fractional["bounds"] == (-3, -4, 26, 15)
    assert fractional["size"] == (29, 19)
    return {
        "identidad": identity,
        "traslaciones": translations,
        "fraccionales": fractional,
    }


def construir_panorama(
    analysis,
    geometry,
    selected_method="distance",
    interpolation=cv2.INTER_LINEAR,
):
    """Deforma, enmascara y combina las tres imágenes.

    Parámetros
    ----------
    analysis : AnalisisCaracteristicas
        Imágenes y orden de la secuencia.
    geometry : ResultadoGeometria
        Homografías laterales hacia el ancla.
    selected_method : str
        Resultado final entre ``overwrite``, ``average`` y ``distance``.
    interpolation : int
        Interpolación de OpenCV usada para las fotografías.

    Retorna
    -------
    ResultadoPanorama
        Bounds, warps, máscaras, pesos y tres combinaciones.
    """
    anchor_index = analysis.anchor_index
    lateral_indices = sorted(
        index for index in analysis.images if index != anchor_index
    )
    if lateral_indices != [0, 2]:
        raise ValueError("El pipeline espera laterales 0 y 2 con ancla 1.")
    bounds = compute_panorama_bounds(
        analysis.images[0],
        analysis.images[anchor_index],
        analysis.images[2],
        geometry.homographies[0],
        geometry.homographies[2],
    )
    final_homographies = {
        0: bounds["H_left_final"],
        anchor_index: bounds["H_anchor_final"],
        2: bounds["H_right_final"],
    }
    warps = {
        index: warp_image_to_canvas(
            analysis.images[index],
            final_homographies[index],
            bounds["size"],
            interpolation,
        )
        for index in sorted(analysis.images)
    }
    masks = {
        index: warp_valid_mask(
            analysis.images[index], final_homographies[index], bounds["size"]
        )
        for index in sorted(analysis.images)
    }
    ordered_warps = [warps[index] for index in sorted(warps)]
    ordered_masks = [masks[index] for index in sorted(masks)]
    weights = {
        index: distance_weight(masks[index]) for index in sorted(masks)
    }
    distance_panorama, denominator = distance_weighted_blend(
        ordered_warps, [weights[index] for index in sorted(weights)]
    )
    blends = {
        "overwrite": stitch_overwrite(ordered_warps, ordered_masks),
        "average": simple_average_blend(ordered_warps, ordered_masks),
        "distance": distance_panorama,
        "denominator": denominator,
    }
    if selected_method not in {"overwrite", "average", "distance"}:
        raise ValueError("Método final de blending no soportado.")
    return ResultadoPanorama(
        analysis=analysis,
        geometry=geometry,
        bounds=bounds,
        warps=warps,
        masks=masks,
        weights=weights,
        blends=blends,
        selected_method=selected_method,
        final=blends[selected_method],
    )


def guardar_panorama(panorama, output_path):
    """Guarda un panorama JPEG y verifica que pueda reabrirse.

    Parámetros
    ----------
    panorama : np.ndarray
        Imagen BGR final.
    output_path : Path
        Ruta de salida.

    Retorna
    -------
    Path
        Ruta validada del archivo escrito.
    """
    path = Path(output_path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"No existe el directorio de salida: {path.parent}")
    success = cv2.imwrite(
        str(path), panorama, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not success:
        raise OSError(f"No se pudo guardar el panorama: {path}")
    loaded = load_image(path)
    if loaded.shape != panorama.shape:
        raise OSError(f"No se pudo validar el panorama guardado: {path}")
    return path
