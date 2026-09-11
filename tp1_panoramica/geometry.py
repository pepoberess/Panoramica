import numpy as np

def normalize_points(pts):
    """
    Normalize 2D points so their centroid is at the origin and their
    average distance to the origin is sqrt(2), per Hartley's method.

    Args:
        pts: (N, 2) array of points.

    Returns:
        pts_norm: (N, 2) array of normalized points.
        T: (3, 3) similarity transform such that pts_norm_hom = T @ pts_hom.
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid

    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale,     0, -scale * centroid[0]],
        [    0, scale, -scale * centroid[1]],
        [    0,     0,                    1]
    ])

    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm_hom = (T @ pts_hom.T).T
    pts_norm = pts_norm_hom[:, :2]

    return pts_norm, T

def dlt(spts, dpts, normalize=False):
    """
    Estimate a homography matrix H such that dst ~ H @ src, using the
    Direct Linear Transformation (DLT) algorithm.

    Args:
        spts: (N, 2) array of points in the source image, N >= 4.
        dpts: (N, 2) array of corresponding points in the destination image.
        normalize: if True, apply Hartley normalization before solving DLT
            and denormalize the resulting H (improves numerical stability).

    Returns:
        H: (3, 3) homography matrix, normalized so H[2, 2] == 1.
    """
    if normalize:
        snorm, T_src = normalize_points(spts)
        dnorm, T_dst = normalize_points(dpts)
        spts, dpts = snorm, dnorm

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
        best_H: (3, 3) homography from the winning sample (4 points only,
            not yet refit on inliers).
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

    return best_H, best_inlier_mask