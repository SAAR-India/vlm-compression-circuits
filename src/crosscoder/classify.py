from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

from .model import SPARCCrossCoder
from .visualize import classify_for_plot, compute_adaptive_rho_thresholds


def compute_decoder_norm_ratio(W_u_dec: torch.Tensor, W_c_dec: torch.Tensor) -> torch.Tensor:
    W_u_norms = W_u_dec.norm(dim=0)
    W_c_norms = W_c_dec.norm(dim=0)
    rho = W_c_norms / (W_u_norms + W_c_norms + 1e-8)
    return rho


def compute_decoder_cosine_similarity(W_u_dec: torch.Tensor, W_c_dec: torch.Tensor) -> torch.Tensor:
    W_u_normalized = F.normalize(W_u_dec, dim=0)
    W_c_normalized = F.normalize(W_c_dec, dim=0)
    theta = (W_u_normalized * W_c_normalized).sum(dim=0)
    return theta


def classify_all_features(crosscoder: SPARCCrossCoder) -> pd.DataFrame:
    """
    Classify features using GMM-based rho thresholds (and fixed theta thresholds).
    feature_classification.csv and all evaluation use these GMM-derived classes.
    """
    decoder_weights = crosscoder.get_decoder_weights()
    W_u_dec = decoder_weights["W_u_dec"]
    W_c_dec = decoder_weights["W_c_dec"]

    rho = compute_decoder_norm_ratio(W_u_dec, W_c_dec)
    theta = compute_decoder_cosine_similarity(W_u_dec, W_c_dec)

    W_u_norms = W_u_dec.norm(dim=0)
    W_c_norms = W_c_dec.norm(dim=0)

    num_features = rho.shape[0]

    # Build initial df with rho, theta (no primary_class yet)
    records = []
    for i in range(num_features):
        records.append({
            "feature_id": i,
            "rho": rho[i].item(),
            "theta": theta[i].item(),
            "W_u_dec_norm": W_u_norms[i].item(),
            "W_c_dec_norm": W_c_norms[i].item(),
            "is_forced_shared": i in crosscoder.forced_shared_indices.tolist(),
        })
    df = pd.DataFrame(records)

    # GMM-based thresholds from rho distribution
    thresh = compute_adaptive_rho_thresholds(df)
    df["primary_class"] = df.apply(
        lambda row: classify_for_plot(row["rho"], row["theta"], thresh), axis=1
    )

    return df


def get_feature_class_counts(classification_df: pd.DataFrame) -> Dict[str, int]:
    return classification_df["primary_class"].value_counts().to_dict()


def get_features_by_class(classification_df: pd.DataFrame, feature_class: str) -> List[int]:
    return classification_df[classification_df["primary_class"] == feature_class]["feature_id"].tolist()


def compute_rho_histogram_data(classification_df: pd.DataFrame, num_bins: int = 50) -> Dict:
    rho_values = classification_df["rho"].values
    hist, bin_edges = torch.histogram(torch.tensor(rho_values), bins=num_bins, range=(0.0, 1.0))
    return {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)],
    }


def compute_threshold_sensitivity(
    classification_df: pd.DataFrame,
    perturbation: float = 0.05,
) -> Dict:
    """
    Sensitivity of class counts to perturbed GMM-based rho thresholds.
    """
    original_counts = get_feature_class_counts(classification_df)
    thresh = compute_adaptive_rho_thresholds(classification_df)

    rho_values = classification_df["rho"].values
    theta_values = classification_df["theta"].values

    perturbed_counts = {}
    for delta in [-perturbation, perturbation]:
        adjusted_thresh = {
            "rho_uncompressed_only": thresh["rho_uncompressed_only"] + delta,
            "rho_compressed_only": thresh["rho_compressed_only"] - delta,
            "rho_shared_low": thresh["rho_shared_low"] + delta,
            "rho_shared_high": thresh["rho_shared_high"] - delta,
        }
        counts = {
            "uncompressed_only": 0,
            "compressed_only": 0,
            "shared_aligned": 0,
            "shared_redirected": 0,
            "shared_intermediate": 0,
            "shared_attenuated": 0,
            "other": 0,
        }
        for rho, theta in zip(rho_values, theta_values):
            c = classify_for_plot(rho, theta, adjusted_thresh)
            counts[c] = counts.get(c, 0) + 1
        perturbed_counts[f"delta_{delta:+.2f}"] = counts

    return {
        "original": original_counts,
        "perturbed": perturbed_counts,
        "perturbation": perturbation,
    }


def save_classification_results(
    classification_df: pd.DataFrame,
    output_path: str,
) -> None:
    classification_df.to_csv(output_path, index=False)


def load_classification_results(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)
