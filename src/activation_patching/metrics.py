"""
Comparison metrics between compressed and uncompressed patching results:
- Jaccard score (overlap of important layers/components)
- Spearman rho (rank correlation of layer/component importance)
- Stability (consistency of importance ordering)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


def _to_binary_topk(scores: np.ndarray, top_k: int = 0.5) -> np.ndarray:
    """Convert importance scores to binary mask (top-k or top 50% by default)."""
    flat = scores.flatten()
    if top_k < 1:
        k = max(1, int(len(flat) * top_k))
    else:
        k = int(top_k)
    threshold = np.partition(flat, -k)[-k] if k <= len(flat) else flat.min()
    return (scores >= threshold).astype(np.float32)


def jaccard_score(
    scores_u: np.ndarray,
    scores_c: np.ndarray,
    top_k: float = 0.5,
) -> float:
    """
    Jaccard similarity between important elements in uncompressed vs compressed.
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    mask_u = _to_binary_topk(scores_u, top_k)
    mask_c = _to_binary_topk(scores_c, top_k)
    intersection = (mask_u * mask_c).sum()
    union = ((mask_u + mask_c) > 0).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def spearman_rho(scores_u: np.ndarray, scores_c: np.ndarray) -> Tuple[float, float]:
    """
    Spearman rank correlation between uncompressed and compressed importance.
    Returns (rho, p-value).
    """
    flat_u = scores_u.flatten()
    flat_c = scores_c.flatten()
    if len(flat_u) != len(flat_c) or len(flat_u) < 2:
        return 0.0, 1.0
    rho, pval = spearmanr(flat_u, flat_c)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(pval)


def stability_score(
    scores_u: np.ndarray,
    scores_c: np.ndarray,
    top_k: int = 5,
) -> float:
    """
    Stability: fraction of top-k elements in uncompressed that remain in top-k of compressed.
    Range [0, 1]; 1 = perfect stability.
    """
    flat_u = scores_u.flatten()
    flat_c = scores_c.flatten()
    n = len(flat_u)
    if n != len(flat_c) or n < top_k:
        return 0.0
    top_u_idx = np.argsort(flat_u)[-top_k:]
    top_c_idx = np.argsort(flat_c)[-top_k:]
    overlap = len(set(top_u_idx) & set(top_c_idx))
    return overlap / top_k


def compute_comparison_metrics(
    metrics_u: Dict,
    metrics_c: Dict,
    components: List[str] = ["mlp", "self_attention", "cross_attention"],
) -> Dict:
    """
    Compute Jaccard, Spearman rho, and stability between uncompressed and compressed.
    metrics_u/c: dict with "per_layer" containing component-wise lists.
    """
    result = {}
    for comp in components:
        lu = metrics_u.get("per_layer", {}).get(comp, [])
        lc = metrics_c.get("per_layer", {}).get(comp, [])
        if not lu or not lc:
            continue
        arr_u = np.array(lu, dtype=np.float64)
        arr_c = np.array(lc, dtype=np.float64)
        if len(arr_u) != len(arr_c):
            min_len = min(len(arr_u), len(arr_c))
            arr_u = arr_u[:min_len]
            arr_c = arr_c[:min_len]
        result[comp] = {
            "jaccard": jaccard_score(arr_u, arr_c),
            "spearman_rho": spearman_rho(arr_u, arr_c)[0],
            "spearman_pval": spearman_rho(arr_u, arr_c)[1],
            "stability": stability_score(arr_u, arr_c, top_k=max(1, len(arr_u) // 4)),
        }
    result["aggregate"] = {
        "jaccard_mean": np.mean([r["jaccard"] for r in result.values()]) if result else 0.0,
        "spearman_mean": np.mean([r["spearman_rho"] for r in result.values()]) if result else 0.0,
        "stability_mean": np.mean([r["stability"] for r in result.values()]) if result else 0.0,
    }
    return result
