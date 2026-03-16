import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from . import config

SHARED_CLASSES = [
    "shared_aligned",
    "shared_redirected",
    "shared_intermediate",
    "shared_attenuated",
]


def compute_feature_sharing_ratio(classification_df: pd.DataFrame) -> float:
    exclusive_classes = ["uncompressed_only", "compressed_only"]

    n_shared = classification_df[classification_df["primary_class"].isin(SHARED_CLASSES)].shape[0]
    n_exclusive = classification_df[classification_df["primary_class"].isin(exclusive_classes)].shape[0]
    
    if n_shared + n_exclusive == 0:
        return 0.0
    
    return n_shared / (n_shared + n_exclusive)


def compute_semantic_stability_score(classification_df: pd.DataFrame) -> float:
    """
    Mean theta over shared features (GMM-based classification).
    Shared = primary_class in shared_aligned, shared_redirected, shared_intermediate, shared_attenuated.
    """
    shared_features = classification_df[
        classification_df["primary_class"].isin(SHARED_CLASSES)
    ]
    if len(shared_features) == 0:
        return float("nan")
    return float(shared_features["theta"].mean())


def compute_counterfactual_sensitivity_shift(merged_df: pd.DataFrame) -> Dict[str, float]:
    results = {}

    for primary_class in merged_df["primary_class"].unique():
        class_df = merged_df[merged_df["primary_class"] == primary_class]
        if len(class_df) > 0 and "cf_shift" in class_df.columns:
            results[primary_class] = class_df["cf_shift"].mean()

    return results


def compute_plan_feature_survival_rate(
    feature_activations: Dict,
    threshold: float = 0.5,
) -> float:
    """
    Plan definition (full_project_plan §5.4): % of original features with high
    correlation in compressed model.
    """
    z_u = feature_activations["z_u"]
    z_c = feature_activations["z_c"]
    if hasattr(z_u, "numpy"):
        z_u = z_u.numpy()
    if hasattr(z_c, "numpy"):
        z_c = z_c.numpy()
    z_u = np.asarray(z_u, dtype=np.float64)
    z_c = np.asarray(z_c, dtype=np.float64)

    num_features = z_u.shape[1]
    survived = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pearsonr ConstantInputWarning for constant features
        for i in range(num_features):
            try:
                corr, _ = pearsonr(z_u[:, i], z_c[:, i])
                if np.isfinite(corr) and corr > threshold:
                    survived += 1
            except (ValueError, RuntimeError):
                pass
    return survived / num_features if num_features > 0 else 0.0


def compute_jaccard_class_distributions(
    classification_a: pd.DataFrame,
    classification_b: pd.DataFrame,
) -> Dict:
    """
    Jaccard similarity of feature class assignments between two configs.
    Per-class: |A_c ∩ B_c| / |A_c ∪ B_c|. Macro-mean over classes.
    """
    # Ensure feature_id alignment (both should have same indices 0..N-1)
    a = classification_a.set_index("feature_id")["primary_class"]
    b = classification_b.set_index("feature_id")["primary_class"]
    all_classes = set(a.unique()) | set(b.unique())
    per_class = {}
    for c in all_classes:
        a_set = set(a[a == c].index.tolist())
        b_set = set(b[b == c].index.tolist())
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        per_class[c] = inter / union if union > 0 else 0.0
    jaccards = list(per_class.values())
    macro_mean = float(np.mean(jaccards)) if jaccards else 0.0
    return {"per_class": per_class, "macro_mean": macro_mean}


def compute_superposition_fraction(superposition_results: Dict) -> float:
    return superposition_results.get("superposition_fraction", 0.0)


def compute_all_primary_metrics(
    classification_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    superposition_results: Dict,
    training_history: Dict,
) -> Dict:
    fsr = compute_feature_sharing_ratio(classification_df)
    sss = compute_semantic_stability_score(classification_df)
    css = compute_counterfactual_sensitivity_shift(merged_df)
    sf = compute_superposition_fraction(superposition_results)
    
    fve_u = training_history["val_fve_u"][-1] if training_history["val_fve_u"] else 0.0
    fve_c = training_history["val_fve_c"][-1] if training_history["val_fve_c"] else 0.0
    dead_neurons = training_history["dead_neurons"][-1] if training_history["dead_neurons"] else 0.0
    l0_u = training_history["l0_u"][-1] if training_history["l0_u"] else 0.0
    l0_c = training_history["l0_c"][-1] if training_history["l0_c"] else 0.0
    
    class_counts = classification_df["primary_class"].value_counts().to_dict()
    
    return {
        "feature_sharing_ratio": fsr,
        "semantic_stability_score": sss,
        "counterfactual_sensitivity_shift": css,
        "superposition_fraction": sf,
        "fve_u": fve_u,
        "fve_c": fve_c,
        "dead_neuron_fraction": dead_neurons,
        "l0_sparsity_u": l0_u,
        "l0_sparsity_c": l0_c,
        "class_counts": class_counts,
        "total_features": len(classification_df),
    }


def test_hypothesis_h1(
    wanda_classification: pd.DataFrame,
    awq_classification: pd.DataFrame,
) -> Dict:
    wanda_u_only = (wanda_classification["primary_class"] == "uncompressed_only").sum()
    wanda_c_only = (wanda_classification["primary_class"] == "compressed_only").sum()
    wanda_attenuated = (wanda_classification["primary_class"] == "shared_attenuated").sum()
    
    awq_u_only = (awq_classification["primary_class"] == "uncompressed_only").sum()
    awq_c_only = (awq_classification["primary_class"] == "compressed_only").sum()
    awq_attenuated = (awq_classification["primary_class"] == "shared_attenuated").sum()
    
    wanda_exclusive = wanda_u_only + wanda_c_only
    awq_exclusive = awq_u_only + awq_c_only
    
    return {
        "wanda_exclusive_count": int(wanda_exclusive),
        "awq_exclusive_count": int(awq_exclusive),
        "wanda_attenuated_count": int(wanda_attenuated),
        "awq_attenuated_count": int(awq_attenuated),
        "hypothesis_supported": awq_attenuated > wanda_attenuated,
        "description": "H1: Wanda produces discrete feature loss, AWQ produces gradual attenuation",
    }


def test_hypothesis_h2(merged_df: pd.DataFrame) -> Dict:
    u_only = merged_df[merged_df["primary_class"] == "uncompressed_only"]
    
    if len(u_only) == 0 or "cf_level_u" not in u_only.columns:
        return {
            "high_cf_count": 0,
            "low_cf_count": 0,
            "ratio": 0.0,
            "hypothesis_supported": False,
            "description": "H2: Visual-evidence features disproportionately lost",
        }
    
    high_cf = (u_only["cf_level_u"] == "high").sum()
    low_cf = (u_only["cf_level_u"] == "low").sum()
    
    ratio = high_cf / (high_cf + low_cf) if (high_cf + low_cf) > 0 else 0.0
    
    return {
        "high_cf_count": int(high_cf),
        "low_cf_count": int(low_cf),
        "ratio": float(ratio),
        "hypothesis_supported": ratio > 0.5,
        "description": "H2: Visual-evidence features disproportionately lost",
    }


def test_hypothesis_h3(
    wanda_superposition: Dict,
    awq_superposition: Dict,
) -> Dict:
    wanda_sf = wanda_superposition.get("superposition_fraction", 0.0)
    awq_sf = awq_superposition.get("superposition_fraction", 0.0)
    
    return {
        "wanda_sf": float(wanda_sf),
        "awq_sf": float(awq_sf),
        "hypothesis_supported": wanda_sf > 0.5 and wanda_sf > awq_sf,
        "description": "H3: Wanda superposition fraction > 50% and > AWQ",
    }


def test_hypothesis_h4(
    cls_classification: pd.DataFrame,
    patch_classification: pd.DataFrame,
) -> Dict:
    fsr_cls = compute_feature_sharing_ratio(cls_classification)
    fsr_patch = compute_feature_sharing_ratio(patch_classification)
    
    return {
        "fsr_cls": float(fsr_cls),
        "fsr_patch": float(fsr_patch),
        "hypothesis_supported": fsr_cls > fsr_patch,
        "description": "H4: CLS tokens have higher FSR than patch tokens",
    }


def test_hypothesis_h5(
    v_classification: pd.DataFrame,
    p_classification: pd.DataFrame,
) -> Dict:
    v_redirected = (v_classification["primary_class"] == "shared_redirected").sum()
    p_redirected = (p_classification["primary_class"] == "shared_redirected").sum()
    
    return {
        "v_redirected_count": int(v_redirected),
        "p_redirected_count": int(p_redirected),
        "hypothesis_supported": p_redirected > v_redirected,
        "description": "H5: Projector has more shared-redirected features than vision encoder",
    }


def test_hypothesis_h6(p_merged_df: pd.DataFrame) -> Dict:
    redirected = p_merged_df[p_merged_df["primary_class"] == "shared_redirected"]
    
    if len(redirected) == 0 or "cf_shift" not in redirected.columns:
        return {
            "mean_cf_shift": 0.0,
            "negative_shift_count": 0,
            "total_redirected": 0,
            "hypothesis_supported": False,
            "description": "H6: Projector redirected features shift from visual to prior",
        }
    
    mean_shift = redirected["cf_shift"].mean()
    negative_count = (redirected["cf_shift"] < 0).sum()
    
    return {
        "mean_cf_shift": float(mean_shift),
        "negative_shift_count": int(negative_count),
        "total_redirected": len(redirected),
        "hypothesis_supported": mean_shift < 0,
        "description": "H6: Projector redirected features shift from visual to prior",
    }


def test_hypothesis_h7(
    v_metrics: Dict,
    p_metrics: Dict,
    vp_metrics: Dict,
) -> Dict:
    fsr_v = v_metrics.get("feature_sharing_ratio", 0.0)
    fsr_p = p_metrics.get("feature_sharing_ratio", 0.0)
    fsr_vp = vp_metrics.get("feature_sharing_ratio", 0.0)
    
    product = fsr_v * fsr_p
    
    return {
        "fsr_v": float(fsr_v),
        "fsr_p": float(fsr_p),
        "fsr_vp": float(fsr_vp),
        "fsr_v_times_p": float(product),
        "hypothesis_supported": fsr_vp < product,
        "description": "H7: FSR(V+P) is sub-additive: FSR(V+P) < FSR(V) × FSR(P)",
    }


def test_hypothesis_h8(
    blip_p_metrics: Dict,
    qwen3vl_p_metrics: Dict,
) -> Dict:
    fsr_blip = blip_p_metrics.get("feature_sharing_ratio", 0.0)
    fsr_qwen3vl = qwen3vl_p_metrics.get("feature_sharing_ratio", 0.0)
    
    return {
        "fsr_blip_p": float(fsr_blip),
        "fsr_qwen3vl_p": float(fsr_qwen3vl),
        "hypothesis_supported": fsr_blip > fsr_qwen3vl,
        "description": "H8: BLIP cross-attention has higher FSR than Qwen3-VL-2B projector",
    }


def compile_all_hypothesis_results(hypothesis_tests: Dict) -> pd.DataFrame:
    records = []
    for h_name, result in hypothesis_tests.items():
        records.append({
            "hypothesis": h_name,
            "supported": result.get("hypothesis_supported", False),
            "description": result.get("description", ""),
            **{k: v for k, v in result.items() if k not in ["hypothesis_supported", "description"]},
        })
    return pd.DataFrame(records)


def compute_decoder_norm_ratio_raw(
    W_u_dec: Union[torch.Tensor, np.ndarray],
    W_c_dec: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Per-feature raw norm ratio: ||W_c[:, i]|| / ||W_u[:, i]||."""
    if isinstance(W_u_dec, torch.Tensor):
        W_u_dec = W_u_dec.cpu().numpy()
    if isinstance(W_c_dec, torch.Tensor):
        W_c_dec = W_c_dec.cpu().numpy()
    W_u_dec = np.asarray(W_u_dec, dtype=np.float64)
    W_c_dec = np.asarray(W_c_dec, dtype=np.float64)
    norms_u = np.linalg.norm(W_u_dec, axis=0)
    norms_c = np.linalg.norm(W_c_dec, axis=0)
    eps = 1e-10
    return norms_c / (norms_u + eps)


def compute_linear_map_summary(
    V_A: np.ndarray,
    V_B: np.ndarray,
    k_min: int = 10,
) -> Optional[Dict]:
    """
    Fit T such that T @ V_A ≈ V_B, compute SVD of restricted map.
    Returns dict with sv_mean, sv_std, condition_number, mean_principal_angle; None if cols < k_min.
    """
    k = V_A.shape[1]
    if k < k_min:
        return None
    V_A = np.asarray(V_A, dtype=np.float64)
    V_B = np.asarray(V_B, dtype=np.float64)
    T = V_B @ V_A.T @ np.linalg.pinv(V_A @ V_A.T)
    T_restricted = T  # T maps from col space of V_A to output; we work in feature subspace
    U, s, Vh = np.linalg.svd(T_restricted)
    cond = float(s[0] / (s[-1] + 1e-12)) if len(s) > 0 and s[-1] > 1e-12 else float("inf")
    T_V_A = T @ V_A
    cos_angles = np.diag(V_A.T @ T_V_A) / (
        np.linalg.norm(V_A, axis=0) * np.linalg.norm(T_V_A, axis=0) + 1e-12
    )
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    principal_angles_rad = np.arccos(np.abs(cos_angles))
    mean_angle_deg = float(np.degrees(principal_angles_rad.mean()))
    return {
        "sv_mean": float(np.mean(s)),
        "sv_std": float(np.std(s)),
        "condition_number": cond,
        "mean_principal_angle_deg": mean_angle_deg,
        "singular_values": [float(x) for x in s[:20]],
    }


def summarize_shared_geometry(
    classification_df: pd.DataFrame,
    W_u_dec: Union[torch.Tensor, np.ndarray],
    W_c_dec: Union[torch.Tensor, np.ndarray],
    k_min: int = 10,
) -> Dict:
    """
    Per shared subclass: distribution stats (rho, theta, angle_deg, norm_ratio_raw)
    and subspace linear-map SVD summaries when subset size >= k_min.
    """
    shared_df = classification_df[
        classification_df["primary_class"].isin(SHARED_CLASSES)
    ].copy()
    if len(shared_df) == 0:
        return {"all_shared": {"n": 0}}

    if isinstance(W_u_dec, torch.Tensor):
        W_u_np = W_u_dec.cpu().numpy()
        W_c_np = W_c_dec.cpu().numpy()
    else:
        W_u_np = np.asarray(W_u_dec, dtype=np.float64)
        W_c_np = np.asarray(W_c_dec, dtype=np.float64)

    norm_ratio_raw = compute_decoder_norm_ratio_raw(W_u_np, W_c_np)
    theta_arr = np.asarray(classification_df["theta"].values, dtype=np.float64)
    theta_clipped = np.clip(theta_arr, -1.0, 1.0)
    angle_deg_arr = np.degrees(np.arccos(np.abs(theta_clipped)))

    extra = pd.DataFrame({
        "feature_id": np.arange(len(classification_df)),
        "norm_ratio_raw": norm_ratio_raw,
        "angle_deg": angle_deg_arr,
    })
    shared_df = shared_df.merge(
        extra[["feature_id", "norm_ratio_raw", "angle_deg"]],
        on="feature_id",
        how="left",
    )

    result: Dict = {}
    classes_to_process: List[str] = list(SHARED_CLASSES) + ["all_shared"]
    for cls in classes_to_process:
        if cls == "all_shared":
            subclass_df = shared_df
        else:
            subclass_df = shared_df[shared_df["primary_class"] == cls]

        n = len(subclass_df)
        row: Dict = {"n": n}
        if n == 0:
            result[cls] = row
            continue

        for col in ["rho", "theta", "angle_deg", "norm_ratio_raw"]:
            if col not in subclass_df.columns:
                continue
            vals = subclass_df[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std())

        ids = subclass_df["feature_id"].astype(int).tolist()
        V_A = W_u_np[:, ids]
        V_B = W_c_np[:, ids]
        lin_summary = compute_linear_map_summary(V_A, V_B, k_min=k_min)
        if lin_summary is not None:
            row["linear_map"] = lin_summary

        result[cls] = row
    return result


def get_shared_features_geometry_df(
    classification_df: pd.DataFrame,
    W_u_dec: Union[torch.Tensor, np.ndarray],
    W_c_dec: Union[torch.Tensor, np.ndarray],
) -> pd.DataFrame:
    """Per-feature geometry for shared features (used for visualization)."""
    shared_df = classification_df[
        classification_df["primary_class"].isin(SHARED_CLASSES)
    ].copy()
    if len(shared_df) == 0:
        return shared_df

    if isinstance(W_u_dec, torch.Tensor):
        W_u_np = W_u_dec.cpu().numpy()
        W_c_np = W_c_dec.cpu().numpy()
    else:
        W_u_np = np.asarray(W_u_dec, dtype=np.float64)
        W_c_np = np.asarray(W_c_dec, dtype=np.float64)

    norm_ratio_raw = compute_decoder_norm_ratio_raw(W_u_np, W_c_np)
    theta_arr = np.asarray(classification_df["theta"].values, dtype=np.float64)
    theta_clipped = np.clip(theta_arr, -1.0, 1.0)
    angle_deg_arr = np.degrees(np.arccos(np.abs(theta_clipped)))

    extra = pd.DataFrame({
        "feature_id": np.arange(len(classification_df)),
        "norm_ratio_raw": norm_ratio_raw,
        "angle_deg": angle_deg_arr,
    })
    return shared_df.merge(
        extra[["feature_id", "norm_ratio_raw", "angle_deg"]],
        on="feature_id",
        how="left",
    )


def save_metrics(metrics: Dict, output_path: str) -> None:
    import json
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(input_path: str) -> Dict:
    import json
    with open(input_path) as f:
        return json.load(f)
