import numpy as np
import matplotlib.pyplot as plt
import cv2

def dlt(spts, dpts):
    """
    Estimate a homography matrix H such that dst ~ H @ src, using the
    Direct Linear Transformation (DLT) algorithm.

    Args:
        spts: (N, 2) array of points in the source image, N >= 4.
        dpts: (N, 2) array of corresponding points in the destination image.

    Returns:
        H: (3, 3) homography matrix, normalized so H[2, 2] == 1.
    """
    n = spts.shape[0]
    A = np.zeros((2 * n, 9))

    for i in range(n):
        x, y = spts[i]
        xp, yp = dpts[i]

        A[2 * i]     = [-x, -y, -1,  0,  0,  0, xp * x, xp * y, xp]
        A[2 * i + 1] = [ 0,  0,  0, -x, -y, -1, yp * x, yp * y, yp]

    # Smallest singular vector of A is the null-space solution to A @ h = 0
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]

    H = h.reshape(3, 3)
    H = H / H[2, 2]

    return H

def compute_reprojection_error(H, spts, dpts):
    """
    Compute the per-point Euclidean reprojection error when mapping
    source points through H and comparing against destination points.

    Args:
        H: (3, 3) homography matrix.
        spts: (N, 2) array of source points.
        dpts: (N, 2) array of corresponding destination points.

    Returns:
        errors: (N,) array of Euclidean distances in pixels.
    """
    n = spts.shape[0]
    shom = np.hstack([spts, np.ones((n, 1))])

    projected_hom = (H @ shom.T).T
    projected = projected_hom[:, :2] / projected_hom[:, 2:3]

    errors = np.linalg.norm(projected - dpts, axis=1)
    return errors

def ransac(spts, dpts, num_iterations=2000, threshold=5.0, seed=None):
    """
    Estimate a homography robust to outliers using RANSAC.

    Args:
        spts: (N, 2) array of source points (noisy correspondences).
        dpts: (N, 2) array of corresponding destination points.
        num_iterations: number of random 4-point samples to try.
        threshold: max reprojection error (pixels) to count as an inlier.
        seed: optional random seed for reproducibility.

    Returns:
        best_H: (3, 3) homography refit by least squares (DLT) over every
            inlier correspondence of the winning sample.
        best_inlier_mask: (N,) boolean array marking inlier correspondences.
    """
    rng = np.random.default_rng(seed)
    n = spts.shape[0]

    best_inlier_count = 0
    best_H = None
    best_inlier_mask = np.zeros(n, dtype=bool)

    for _ in range(num_iterations):
        sample_idx = rng.choice(n, size=4, replace=False)
        H_candidate = dlt(spts[sample_idx], dpts[sample_idx])

        errors = compute_reprojection_error(H_candidate, spts, dpts)
        inlier_mask = errors < threshold
        inlier_count = inlier_mask.sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_H = H_candidate
            best_inlier_mask = inlier_mask

    if best_H is None:
        raise RuntimeError("RANSAC no encontró ningún modelo válido.")

    # Refit la homografia con todos los inliers de la mejor muestra.
    best_H = dlt(spts[best_inlier_mask], dpts[best_inlier_mask])

    # La mascara de inliers usada para el refit corresponde a la H de la
    # muestra de 4 puntos, no a la H final. Se recalculan errores e inliers
    # con la H final para que ambos sean consistentes entre si.
    errors = compute_reprojection_error(best_H, spts, dpts)
    best_inlier_mask = errors < threshold

    return best_H, best_inlier_mask