"""
Image processing pipeline:
  binarize → morphological close → skeletonize → bridge endpoints
  → prune to longest branch → dilate back to thickness.
"""

import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize
from scipy.ndimage import label
from scipy.stats import norm


# ============================================================
#  Binarization
# ============================================================

def binarize(gray: np.ndarray, params: dict) -> np.ndarray:
    """Binarize a grayscale image.

    params keys:
        bin_method   – "mean" | "otsu" | "adaptive" | "fixed"
        bin_ratio    – for "mean": upper quantile (default 0.25)
        bin_threshold – for "fixed": raw threshold (default 128)
        adaptive_block – for "adaptive": block size (default 31)
        adaptive_c     – for "adaptive": subtracted constant (default 2)
    """
    method = params.get("bin_method", "mean")

    if method == "mean":
        ratio = params.get("bin_ratio", 0.25)
        mean = float(gray.mean())
        std  = float(gray.std())
        t = mean + std * norm.ppf(1.0 - ratio)
        t = np.clip(t, gray.min(), gray.max())
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        block = params.get("adaptive_block", 31)
        c     = params.get("adaptive_c", 2)
        # ensure odd block size
        if block % 2 == 0:
            block += 1
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, c,
        )

    elif method == "fixed":
        t = params.get("bin_threshold", 128)
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    else:
        raise ValueError(f"unknown binarize method: {method}")

    return binary


# ============================================================
#  Morphological close (connect nearby regions)
# ============================================================

def morphological_close(binary: np.ndarray, params: dict) -> np.ndarray:
    """Close small gaps to connect nearby foreground regions."""
    ks = params.get("close_kernel", 5)
    it = params.get("close_iter", 2)
    if ks % 2 == 0:
        ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    closed = binary
    for _ in range(it):
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    return closed


# ============================================================
#  Skeletonize + endpoint bridging
# ============================================================

def _find_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (y, x) for skeleton pixels with exactly 1 neighbour."""
    h, w = skel.shape
    endpoints = []
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        cnt = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    cnt += 1
        if cnt == 1:
            endpoints.append((y, x))
    return endpoints


def _bridge_skeleton(skel: np.ndarray, max_dist: int) -> np.ndarray:
    """Draw straight lines between nearby endpoints, then re-skeletonize."""
    eps = _find_endpoints(skel)
    if len(eps) < 2:
        return skel

    bridge_mask = np.zeros_like(skel)
    n = len(eps)
    for i in range(n):
        yi, xi = eps[i]
        for j in range(i + 1, n):
            yj, xj = eps[j]
            d = np.sqrt((yi - yj) ** 2 + (xi - xj) ** 2)
            if d <= max_dist:
                cv2.line(bridge_mask, (xi, yi), (xj, yj), 255, 1)

    if bridge_mask.any():
        combined = np.maximum(skel, bridge_mask)
        return skeletonize(combined.astype(bool)).astype(np.uint8) * 255
    return skel


def skeletonize_and_bridge(binary: np.ndarray, params: dict) -> np.ndarray:
    """Skeletonize binary mask, optionally bridge nearby endpoints."""
    skel = skeletonize(binary.astype(bool)).astype(np.uint8) * 255

    if params.get("bridge", False):
        skel = _bridge_skeleton(skel, params.get("bridge_dist", 8))

    return skel


# ============================================================
#  Prune to longest branch
# ============================================================

def _bfs_farthest(
    skel: np.ndarray, start: tuple[int, int]
) -> tuple[tuple[int, int], dict]:
    """BFS from *start*; return (farthest_node, parent_dict)."""
    h, w = skel.shape
    q = deque([start])
    visited = {start}
    parent = {start: None}
    farthest = start

    while q:
        node = q.popleft()
        farthest = node
        y, x = node
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    nb = (ny, nx)
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        q.append(nb)
    return farthest, parent


def _longest_path(skel: np.ndarray) -> np.ndarray:
    """Return a skeleton containing only the longest simple path."""
    eps = _find_endpoints(skel)
    if len(eps) < 2:
        return skel  # nothing to prune

    # BFS from first endpoint → find farthest
    a = eps[0]
    b, parent1 = _bfs_farthest(skel, a)
    # BFS from b → find actual farthest + parent chain
    c, parent2 = _bfs_farthest(skel, b)

    # Reconstruct path c → ... → b
    path_pixels = set()
    node = c
    while node is not None:
        path_pixels.add(node)
        node = parent2.get(node)  # parent2 is from b

    out = np.zeros_like(skel)
    for y, x in path_pixels:
        out[y, x] = 255
    return out


def prune_skeleton(skel: np.ndarray, params: dict) -> np.ndarray:
    """Keep only the longest branch of the skeleton."""
    method = params.get("prune_method", "longest_path")

    if method == "longest_path":
        return _longest_path(skel)

    elif method == "largest_component":
        labeled, ncomp = label(skel > 0, structure=np.ones((3, 3), dtype=bool))
        if ncomp == 0:
            return skel
        sizes = [(labeled == i).sum() for i in range(1, ncomp + 1)]
        best = np.argmax(sizes) + 1
        out = np.zeros_like(skel)
        out[labeled == best] = 255
        return out

    else:
        raise ValueError(f"unknown prune method: {method}")


# ============================================================
#  Estimate thickness
# ============================================================

def estimate_thickness(gray_or_binary: np.ndarray) -> float:
    """Estimate average half-thickness (radius) of foreground regions.

    Uses distance transform on a binarized version.
    Returns the ~90th percentile distance from foreground pixels to the nearest
    boundary, which better captures the true thickness of trabecular structures
    than the median (which is easily pulled down to 1 px by edge pixels).
    """
    # Binarize if needed
    if gray_or_binary.dtype == np.uint8 and set(np.unique(gray_or_binary)) <= {0, 255}:
        binary = gray_or_binary
    else:
        binary = binarize(gray_or_binary, {"bin_method": "mean", "bin_ratio": 0.25})

    if not binary.any():
        return 2.0

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    vals = dist[binary > 0]
    if len(vals) == 0:
        return 2.0
    # Use ~90th percentile instead of median to avoid being pulled down
    # by the large number of edge pixels (dist=1).
    return float(np.percentile(vals, 90))


# ============================================================
#  Dilate skeleton back to thickness
# ============================================================

def _smooth_skeleton_path(skel: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 1-px skeleton path by applying moving average to coordinates,
    then re-rasterizing. This removes jagged 'staircase' artifacts."""
    ys, xs = np.where(skel)
    if len(ys) < window * 2:
        return skel  # too short to smooth meaningfully

    # Trace the path: start from an endpoint
    from collections import deque
    h, w = skel.shape
    eps = _find_endpoints(skel)
    if not eps:
        return skel

    start = eps[0]
    # BFS to order pixels along the path
    q = deque([start])
    visited = {start}
    ordered = [start]
    parent = {start: None}
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    nb = (ny, nx)
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = (y, x)
                        q.append(nb)
                        ordered.append(nb)

    if len(ordered) < window:
        return skel

    # Convert to numpy arrays for smoothing
    pts = np.array(ordered, dtype=np.float64)  # (N, 2)  each row = (y, x)
    N = len(pts)

    # Moving average with reflection padding to preserve endpoints
    kernel = np.ones(window) / window
    # Pad
    pad = window // 2
    ys_pad = np.pad(pts[:, 0], pad, mode='reflect')
    xs_pad = np.pad(pts[:, 1], pad, mode='reflect')
    ys_smooth = np.convolve(ys_pad, kernel, mode='valid')
    xs_smooth = np.convolve(xs_pad, kernel, mode='valid')

    # Re-rasterize
    out = np.zeros_like(skel)
    for y, x in zip(np.round(ys_smooth).astype(int), np.round(xs_smooth).astype(int)):
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = 255

    # If smoothing broke connectivity, fall back to original
    if not out.any():
        return skel

    return out


def dilate_skeleton(
    skel: np.ndarray,
    params: dict,
    original_gray: np.ndarray | None = None,
) -> np.ndarray:
    """Dilate the 1-px skeleton back to original thickness.

    If params["use_distance"] is True, dilate iterations are derived from the
    distance transform of *original_gray* (original grayscale image, NOT binarized).
    This avoids using "ossified" (骨化后的) data for thickness estimation.
    Otherwise use fixed kernel size + iteration count.
    """
    ks = params.get("dilate_kernel", 3)
    if ks % 2 == 0:
        ks += 1

    if params.get("use_distance", False):
        ref = original_gray if original_gray is not None else skel
        radius = estimate_thickness(ref)
        iterations = max(1, int(np.ceil(radius)))
    else:
        iterations = params.get("dilate_iter", 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = skel
    for _ in range(iterations):
        dilated = cv2.dilate(dilated, kernel)
    return dilated


# ============================================================
#  Full pipeline
# ============================================================

def process_pipeline(gray: np.ndarray, params: dict) -> np.ndarray:
    """Run the complete pipeline on a grayscale image.

    Steps:
      1. binarize
      2. morphological close (connect nearby regions)
      3. skeletonize + optionally bridge endpoints
      4. prune to longest branch
      5. smooth the longest path (remove jaggies)
      6. dilate back to approximate original thickness
      7. smooth edges of dilated result
    """
    # 1. Binarize
    binary = binarize(gray, params)

    # 2. Connect nearby regions
    closed = morphological_close(binary, params)

    # 3. Skeletonize + bridge
    skel = skeletonize_and_bridge(closed, params)

    if not skel.any():
        return binary  # fallback

    # 4. Prune to longest
    pruned = prune_skeleton(skel, params)

    # 5. Smooth the skeleton path (remove jaggies before dilation)
    pruned = _smooth_skeleton_path(pruned)

    # 6. Dilate back (use original grayscale for thickness estimation, not binarized)
    result = dilate_skeleton(pruned, params, original_gray=gray)

    # 7. Light Gaussian smoothing to clean up edges of dilated result
    if result.any():
        blurred = cv2.GaussianBlur(result.astype(np.float32), (0, 0), sigmaX=1.0)
        _, result = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
        result = result.astype(np.uint8)

    return result
