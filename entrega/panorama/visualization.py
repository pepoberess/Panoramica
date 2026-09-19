"""Resúmenes y visualizaciones compactas para el notebook final."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .pipeline import MATCHING_LABELS
from .warping import warp_image_to_canvas


def _rgb(image):
    """Convierte BGR a RGB para mostrar con Matplotlib."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_preview(image, maximum_width=1500, maximum_height=700):
    """Reduce una visualización conservando su relación de aspecto."""
    scale = min(
        maximum_width / image.shape[1],
        maximum_height / image.shape[0],
        1.0,
    )
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def imprimir_resumen_caracteristicas(analysis):
    """Imprime dimensiones y cantidades antes y después de A-NMS."""
    print(f"Dataset: {analysis.nombre}; ancla: {analysis.anchor_index}")
    for index in sorted(analysis.images):
        image = analysis.images[index]
        print(
            f"Imagen {index}: {image.shape[1]} x {image.shape[0]} px; "
            f"{len(analysis.keypoints[index])} keypoints → "
            f"{len(analysis.anms_keypoints[index])} después de A-NMS"
        )


def mostrar_tripleta(analysis):
    """Muestra las tres imágenes e identifica el ancla."""
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    for index, axis in zip(sorted(analysis.images), axes):
        role = "ancla" if index == analysis.anchor_index else "lateral"
        axis.imshow(_rgb(analysis.images[index]))
        axis.set_title(f"{analysis.nombre} {index} ({role})")
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def mostrar_keypoints_y_anms(analysis):
    """Compara los puntos SIFT iniciales con la selección A-NMS."""
    _, axes = plt.subplots(2, 3, figsize=(16, 9))
    for column, index in enumerate(sorted(analysis.images)):
        for row, (keypoints, stage) in enumerate(
            [
                (analysis.keypoints[index], "SIFT"),
                (analysis.anms_keypoints[index], "SIFT + A-NMS"),
            ]
        ):
            axis = axes[row, column]
            axis.imshow(_rgb(analysis.images[index]))
            points = np.array([keypoint.pt for keypoint in keypoints])
            if len(points):
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=8 if row == 0 else 13,
                    facecolors="none",
                    edgecolors="lime",
                    linewidths=0.45,
                )
            axis.set_title(f"{stage}: imagen {index} ({len(keypoints)})")
            axis.axis("off")
    plt.tight_layout()
    plt.show()


def mostrar_comparacion_matching(analysis, maximum_matches=40):
    """Muestra Lowe, cross-check y la combinación en ambos pares."""
    source_indices = sorted(analysis.matches)
    _, axes = plt.subplots(len(source_indices), 3, figsize=(18, 9))
    axes = np.atleast_2d(axes)
    for row, source_index in enumerate(source_indices):
        for column, method in enumerate(MATCHING_LABELS):
            matches = analysis.matching_experiments[method][source_index]
            selected = sorted(
                matches, key=lambda match: match.distance
            )[:maximum_matches]
            preview = cv2.drawMatches(
                analysis.images[source_index],
                analysis.anms_keypoints[source_index],
                analysis.images[analysis.anchor_index],
                analysis.anms_keypoints[analysis.anchor_index],
                selected,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            axes[row, column].imshow(_rgb(_resize_preview(preview)))
            axes[row, column].set_title(
                f"{MATCHING_LABELS[method]}\n{source_index} → "
                f"{analysis.anchor_index}: {len(matches)}"
            )
            axes[row, column].axis("off")
    plt.tight_layout()
    plt.show()


def mostrar_matching_seleccionado(analysis, maximum_matches=50):
    """Muestra solamente la política de matching elegida en ambos pares."""
    source_indices = sorted(analysis.matches)
    _, axes = plt.subplots(1, len(source_indices), figsize=(16, 6))
    axes = np.atleast_1d(axes)
    for axis, source_index in zip(axes, source_indices):
        matches = analysis.matches[source_index]
        selected = sorted(
            matches, key=lambda match: match.distance
        )[:maximum_matches]
        preview = cv2.drawMatches(
            analysis.images[source_index],
            analysis.anms_keypoints[source_index],
            analysis.images[analysis.anchor_index],
            analysis.anms_keypoints[analysis.anchor_index],
            selected,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        axis.imshow(_rgb(_resize_preview(preview)))
        axis.set_title(
            f"{MATCHING_LABELS[analysis.selected_method]}: "
            f"{source_index} → {analysis.anchor_index} ({len(matches)})"
        )
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def imprimir_evaluacion_matching(rows):
    """Imprime la evaluación geométrica de los métodos de matching."""
    print("Método                    | Par   | Matches | Inliers | % inliers | Error medio")
    print("-" * 83)
    for row in rows:
        error = (
            "n/d"
            if not np.isfinite(row["mean_error"])
            else f"{row['mean_error']:.3f} px"
        )
        print(
            f"{MATCHING_LABELS[row['method']]:<25} | "
            f"{row['source_index']} → 1 | {row['matches']:>7} | "
            f"{row['inliers']:>7} | {row['percentage']:>8.1f}% | {error}"
        )




def mostrar_correspondencias_manuales(analysis, correspondences, udesa=False):
    """Grafica los cuatro puntos manuales en laterales y ancla."""
    source_indices = sorted(correspondences)
    _, axes = plt.subplots(len(source_indices), 2, figsize=(12, 10))
    axes = np.atleast_2d(axes)
    for row, source_index in enumerate(source_indices):
        source_points, anchor_points = correspondences[source_index]
        for axis, image, points, title in [
            (
                axes[row, 0],
                analysis.images[source_index],
                source_points,
                f"Puntos manuales: Udesa {source_index}" if udesa
                else f"Puntos manuales: Cuadro {source_index}",
            ),
            (
                axes[row, 1],
                analysis.images[analysis.anchor_index],
                anchor_points,
                "Correspondencias en el ancla",
            ),
        ]:
            axis.imshow(_rgb(image))
            axis.scatter(
                points[:, 0], points[:, 1], c="red", s=55, marker="x"
            )
            for point_index, (x_coord, y_coord) in enumerate(points):
                axis.annotate(
                    str(point_index),
                    (x_coord, y_coord),
                    color="yellow",
                    fontsize=12,
                    weight="bold",
                )
            axis.set_title(title)
            axis.axis("off")
    plt.tight_layout()
    plt.show()


def imprimir_homografias_manuales(results, udesa=False):
    """Imprime las matrices DLT y el error sobre los cuatro puntos."""
    for source_index in sorted(results):
        errors = results[source_index]["errors"]
        if udesa:
            print(f"Udesa {source_index} → 1")
        else:
            print(f"Cuadro {source_index} → 1")
        print(results[source_index]["H"])
        print(
            f"Error en los cuatro puntos: media {errors.mean():.6f} px; "
            f"máximo {errors.max():.6f} px\n"
        )


def mostrar_alineacion_manual(analysis, correspondences, results):
    """Superpone cada lateral deformada con el ancla en la zona elegida."""
    source_indices = sorted(correspondences)
    _, axes = plt.subplots(1, len(source_indices), figsize=(15, 7))
    axes = np.atleast_1d(axes)
    anchor = analysis.images[analysis.anchor_index]
    anchor_size = (anchor.shape[1], anchor.shape[0])
    for axis, source_index in zip(axes, source_indices):
        homography = results[source_index]["H"]
        warped = warp_image_to_canvas(
            analysis.images[source_index],
            homography,
            anchor_size,
            cv2.INTER_LINEAR,
        )
        valid = warp_image_to_canvas(
            np.full(analysis.images[source_index].shape[:2], 255, dtype=np.uint8),
            homography,
            anchor_size,
            cv2.INTER_NEAREST,
        ) > 0
        diagnostic = anchor.copy()
        diagnostic[valid] = cv2.addWeighted(
            warped[valid], 0.5, anchor[valid], 0.5, 0
        )
        anchor_points = correspondences[source_index][1]
        margin = 350
        x_start = max(int(anchor_points[:, 0].min()) - margin, 0)
        x_end = min(int(anchor_points[:, 0].max()) + margin, anchor.shape[1])
        y_start = max(int(anchor_points[:, 1].min()) - margin, 0)
        y_end = min(int(anchor_points[:, 1].max()) + margin, anchor.shape[0])
        axis.imshow(_rgb(diagnostic[y_start:y_end, x_start:x_end]))
        axis.set_title(f"DLT manual: {source_index} → {analysis.anchor_index}")
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def imprimir_comparacion_thresholds(rows):
    """Imprime inliers y errores para cada umbral de RANSAC."""
    print("Par   | Umbral | Inliers/total | % inliers | Error medio | Mediana")
    print("-" * 75)
    for row in rows:
        print(
            f"{row['source_index']} → 1 | {row['threshold']:>6.1f} | "
            f"{row['inliers']:>3}/{row['total']:<3}      | "
            f"{row['percentage']:>8.1f}% | {row['mean_error']:>8.3f} px | "
            f"{row['median_error']:>7.3f} px"
        )


def imprimir_resumen_geometria(analysis, geometry):
    """Imprime homografías, inliers y error final de reproyección."""
    for source_index in sorted(geometry.homographies):
        mask = geometry.inlier_masks[source_index]
        errors = geometry.errors[source_index]
        inlier_errors = errors[mask]
        print(f"{analysis.nombre}: {source_index} → {analysis.anchor_index}")
        print(geometry.homographies[source_index])
        print(
            f"Inliers: {mask.sum()}/{len(mask)} ({100 * mask.mean():.1f}%); "
            f"error medio {inlier_errors.mean():.3f} px; "
            f"mediana {np.median(inlier_errors):.3f} px; "
            f"máximo {inlier_errors.max():.3f} px\n"
        )


def mostrar_inliers_outliers(analysis, geometry):
    """Dibuja en verde los inliers y en rojo los outliers."""
    source_indices = sorted(geometry.inlier_masks)
    _, axes = plt.subplots(len(source_indices), 1, figsize=(16, 10))
    axes = np.atleast_1d(axes)
    for axis, source_index in zip(axes, source_indices):
        mask = geometry.inlier_masks[source_index]
        matches = analysis.matches[source_index]
        inliers = [match for match, valid in zip(matches, mask) if valid]
        outliers = [match for match, valid in zip(matches, mask) if not valid]
        preview = cv2.drawMatches(
            analysis.images[source_index],
            analysis.anms_keypoints[source_index],
            analysis.images[analysis.anchor_index],
            analysis.anms_keypoints[analysis.anchor_index],
            outliers,
            None,
            matchColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        preview = cv2.drawMatches(
            analysis.images[source_index],
            analysis.anms_keypoints[source_index],
            analysis.images[analysis.anchor_index],
            analysis.anms_keypoints[analysis.anchor_index],
            inliers,
            preview,
            matchColor=(0, 255, 0),
            flags=(
                cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
                | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            ),
        )
        axis.imshow(_rgb(_resize_preview(preview, 1600, 750)))
        axis.set_title(
            f"{source_index} → {analysis.anchor_index}: "
            f"verde={len(inliers)}; rojo={len(outliers)}"
        )
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def imprimir_pruebas_limites(results):
    """Resume las pruebas controladas del cálculo de canvas."""
    for name, result in results.items():
        print(
            f"{name.capitalize()}: bounds={result['bounds']}; "
            f"tamaño={result['size']}"
        )
    print("Pruebas controladas de límites: OK")


def mostrar_limites_panorama(panorama):
    """Grafica esquinas transformadas y el rectángulo del canvas."""
    width, height = panorama.bounds["size"]
    _, axis = plt.subplots(figsize=(10, 6))
    for name, color in [
        ("left", "tab:orange"),
        ("anchor", "tab:blue"),
        ("right", "tab:green"),
    ]:
        corners = panorama.bounds["corners_final"][name]
        contour = np.vstack([corners, corners[0]])
        axis.plot(contour[:, 0], contour[:, 1], "o-", color=color, label=name)
    axis.add_patch(
        Rectangle(
            (0, 0),
            width,
            height,
            fill=False,
            linestyle="--",
            edgecolor="black",
            linewidth=2,
            label="canvas",
        )
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_title(f"Canvas óptimo: {width} × {height}")
    axis.set_xlabel("x (columnas)")
    axis.set_ylabel("y (filas)")
    axis.legend()
    plt.tight_layout()
    plt.show()
    print("Bounds (xmin, ymin, xmax, ymax):", panorama.bounds["bounds"])
    print("Tamaño (ancho, alto):", panorama.bounds["size"])
    print("Traslación T:\n", panorama.bounds["T"])


def comparar_interpolaciones(panorama, source_index=0, crop_radius=90):
    """Compara NEAREST y LINEAR en el mismo detalle transformado.

    Retorna
    -------
    dict
        Diferencia media y porcentaje de píxeles distintos.
    """
    homography_key = "H_left_final" if source_index == 0 else "H_right_final"
    homography = panorama.bounds[homography_key]
    image = panorama.analysis.images[source_index]
    nearest = warp_image_to_canvas(
        image, homography, panorama.bounds["size"], cv2.INTER_NEAREST
    )
    linear = panorama.warps[source_index]
    center_source = np.array(
        [image.shape[1] / 2, image.shape[0] / 2, 1.0], dtype=float
    )
    center_homogeneous = homography @ center_source
    center = np.rint(
        center_homogeneous[:2] / center_homogeneous[2]
    ).astype(int)
    width, height = panorama.bounds["size"]
    x_start = max(0, center[0] - crop_radius)
    x_end = min(width, center[0] + crop_radius)
    y_start = max(0, center[1] - crop_radius)
    y_end = min(height, center[1] + crop_radius)
    crop_nearest = nearest[y_start:y_end, x_start:x_end]
    crop_linear = linear[y_start:y_end, x_start:x_end]
    difference = np.abs(
        crop_nearest.astype(np.float32) - crop_linear.astype(np.float32)
    )
    metrics = {
        "mean_absolute_difference": float(difference.mean()),
        "different_pixels_percentage": float(
            100.0 * np.any(difference > 0, axis=2).mean()
        ),
    }
    _, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, crop, title in [
        (axes[0], crop_nearest, "INTER_NEAREST"),
        (axes[1], crop_linear, "INTER_LINEAR"),
    ]:
        axis.imshow(_rgb(crop), interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    plt.show()
    print(
        "Diferencia absoluta media: "
        f"{metrics['mean_absolute_difference']:.3f} niveles por canal"
    )
    print(
        "Píxeles distintos: "
        f"{metrics['different_pixels_percentage']:.1f}%"
    )
    return metrics


def mostrar_warps_mascaras_y_pesos(panorama):
    """Muestra warps, máscaras binarias y pesos de distancia."""
    indices = sorted(panorama.warps)
    _, axes = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    maximum_weight = max(weight.max() for weight in panorama.weights.values())
    heatmap = None
    for column, index in enumerate(indices):
        axes[0, column].imshow(_rgb(panorama.warps[index]))
        axes[0, column].set_title(f"Warp {index}")
        axes[1, column].imshow(
            panorama.masks[index], cmap="gray", vmin=0, vmax=255
        )
        axes[1, column].set_title(f"Máscara {index}")
        heatmap = axes[2, column].imshow(
            panorama.weights[index],
            cmap="inferno",
            vmin=0,
            vmax=maximum_weight,
        )
        axes[2, column].set_title(f"Peso {index}")
        for row in range(3):
            axes[row, column].axis("off")
    plt.colorbar(
        heatmap,
        ax=axes[2, :],
        shrink=0.75,
        label="Distancia al exterior (px)",
    )
    plt.show()


def _diagnostic_crops(panorama, radius=220):
    """Elige recortes alrededor de la mediana de los inliers."""
    width, height = panorama.bounds["size"]
    translation = panorama.bounds["T"]
    crops = {}
    for source_index in sorted(panorama.geometry.inlier_masks):
        mask = panorama.geometry.inlier_masks[source_index]
        anchor_points = panorama.analysis.matched_points[source_index]["anchor"][mask]
        center_anchor = np.median(anchor_points, axis=0)
        center_canvas = center_anchor + translation[:2, 2]
        center_x, center_y = np.rint(center_canvas).astype(int)
        crops[source_index] = (
            max(0, center_x - radius),
            min(width, center_x + radius),
            max(0, center_y - radius),
            min(height, center_y + radius),
        )
    return crops


def mostrar_diagnostico_alineamiento(panorama):
    """Superpone cada lateral con el ancla en una zona con inliers."""
    anchor_index = panorama.analysis.anchor_index
    crops = _diagnostic_crops(panorama)
    _, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, source_index in enumerate(sorted(crops)):
        x_start, x_end, y_start, y_end = crops[source_index]
        lateral = panorama.warps[source_index][y_start:y_end, x_start:x_end]
        anchor = panorama.warps[anchor_index][y_start:y_end, x_start:x_end]
        overlap = (
            (panorama.masks[source_index][y_start:y_end, x_start:x_end] > 0)
            & (panorama.masks[anchor_index][y_start:y_end, x_start:x_end] > 0)
        )
        lateral_visible = lateral.copy()
        anchor_visible = anchor.copy()
        lateral_visible[~overlap] = 0
        anchor_visible[~overlap] = 0
        diagnostic = cv2.addWeighted(
            lateral_visible, 0.5, anchor_visible, 0.5, 0
        )
        for axis, image, title in zip(
            axes[row],
            [lateral_visible, anchor_visible, diagnostic],
            ["Lateral", "Ancla", "Superposición 50/50"],
        ):
            axis.imshow(_rgb(image), interpolation="nearest")
            axis.set_title(f"Par {source_index} → {anchor_index}: {title}")
            axis.axis("off")
    plt.tight_layout()
    plt.show()
    return crops


def mostrar_comparacion_blending(panorama, crops=None):
    """Compara overwrite, promedio y pesos de distancia."""
    selected_crops = _diagnostic_crops(panorama) if crops is None else crops
    methods = [
        ("overwrite", "Overwrite"),
        ("average", "Promedio simple"),
        ("distance", "Distance Transform"),
    ]
    _, axes = plt.subplots(3, 3, figsize=(18, 14))
    for column, (method, label) in enumerate(methods):
        image = panorama.blends[method]
        axes[0, column].imshow(_rgb(image), interpolation="nearest")
        axes[0, column].set_title(f"{label}: completo")
        axes[0, column].axis("off")
        for row, source_index in enumerate(sorted(selected_crops), start=1):
            x_start, x_end, y_start, y_end = selected_crops[source_index]
            axes[row, column].imshow(
                _rgb(image[y_start:y_end, x_start:x_end]),
                interpolation="nearest",
            )
            axes[row, column].set_title(f"{label}: par {source_index} → 1")
            axes[row, column].axis("off")
    plt.tight_layout()
    plt.show()


def mostrar_panorama_final(panorama, title):
    """Muestra el panorama elegido e imprime sus propiedades."""
    plt.figure(figsize=(16, 8))
    plt.imshow(_rgb(panorama.final), interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    print("Método final:", panorama.selected_method)
    print("Shape:", panorama.final.shape, "dtype:", panorama.final.dtype)
