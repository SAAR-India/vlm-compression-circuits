import argparse
import sys
from pathlib import Path

from . import config
from .utils import (
    get_activations_dir,
    get_checkpoint_dir,
    get_features_dir,
    get_metrics_dir,
    get_plots_dir,
    get_results_dir,
    load_activations,
    load_json,
    save_activations,
    save_json,
    set_seed,
)


def run_extract(model: str, method: str, component: str, token_type: str):
    from .activations import extract_activations_for_config, extract_activations_for_vp_config
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION: {model}/{method}/{component}/{token_type}")
    print(f"{'='*60}")
    
    results_dir = get_results_dir(model, method, component, token_type)
    activations_dir = get_activations_dir(results_dir)
    
    if component == "V_P":
        v_path = activations_dir / "activations_V.pt"
        p_path = activations_dir / "activations_P.pt"
        if v_path.exists() and p_path.exists():
            print("Activations already exist (V and P), skipping extraction.")
            return
        v_result, p_result = extract_activations_for_vp_config(model, method)
        save_activations(v_result, v_path)
        save_activations(p_result, p_path)
        print(f"Saved V activations: {v_path}")
        print(f"Saved P activations: {p_path}")
    else:
        out_path = activations_dir / "activations.pt"
        if out_path.exists():
            print(f"Activations already exist at {out_path}, skipping extraction.")
            return
        result = extract_activations_for_config(model, method, component, token_type)
        save_activations(result, out_path)
        print(f"Saved activations: {out_path}")
    
    print("Extraction complete!")


def run_train(model: str, method: str, component: str, token_type: str):
    from .train import train_crosscoder

    print(f"\n{'='*60}")
    print(f"TRAINING: {model}/{method}/{component}/{token_type}")
    print(f"{'='*60}")

    results_dir = get_results_dir(model, method, component, token_type)
    activations_dir = get_activations_dir(results_dir)
    checkpoint_dir = get_checkpoint_dir(results_dir)
    if (checkpoint_dir / "final.pt").exists():
        print(f"Checkpoint already exists at {checkpoint_dir / 'final.pt'}, skipping training.")
        return None

    if component == "V_P":
        activations_path = activations_dir / f"activations_{token_type.upper() if token_type in ['cls', 'patch'] else 'V'}.pt"
        if not activations_path.exists():
            activations_path = activations_dir / "activations_V.pt"
    else:
        activations_path = activations_dir / "activations.pt"

    activations_data = load_activations(activations_path)

    train_result = train_crosscoder(
        activations_data=activations_data,
        model_name=model,
        method=method,
        component=component,
        token_type=token_type,
    )
    
    print("Training complete!")
    return train_result


def run_analyze(model: str, method: str, component: str, token_type: str):
    from .classify import classify_all_features, save_classification_results
    from .counterfactual import (
        classify_cf_level,
        compute_cf_shift_by_class,
        compute_counterfactual_sensitivity,
        identify_visual_evidence_features,
        merge_classification_with_cf,
        save_cf_results,
    )
    from .metrics import compute_all_primary_metrics, save_metrics
    from .superposition import analyze_all_compressed_only_features, save_superposition_results
    from .train import compute_all_feature_activations, load_trained_crosscoder

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {model}/{method}/{component}/{token_type}")
    print(f"{'='*60}")

    results_dir = get_results_dir(model, method, component, token_type)
    activations_dir = get_activations_dir(results_dir)
    features_dir = get_features_dir(results_dir)
    metrics_dir = get_metrics_dir(results_dir)

    aggregate_path = metrics_dir / "aggregate_metrics.json"
    if aggregate_path.exists():
        print(f"Analysis outputs already exist at {aggregate_path}, skipping analysis.")
        return

    if component == "V_P":
        activations_path = activations_dir / "activations_V.pt"
    else:
        activations_path = activations_dir / "activations.pt"

    activations_data = load_activations(activations_path)
    
    print("Loading trained cross-coder...")
    crosscoder = load_trained_crosscoder(model, method, component, token_type)
    
    print("Computing feature activations...")
    feature_activations = compute_all_feature_activations(crosscoder, activations_data)
    
    print("Classifying features...")
    classification_df = classify_all_features(crosscoder)
    save_classification_results(classification_df, features_dir / "feature_classification.csv")
    
    print("Computing counterfactual sensitivity...")
    cf_scores_df = compute_counterfactual_sensitivity(feature_activations)
    cf_scores_df = classify_cf_level(cf_scores_df)
    save_cf_results(cf_scores_df, features_dir / "counterfactual_scores.csv")
    
    merged_df = merge_classification_with_cf(classification_df, cf_scores_df)
    merged_df.to_csv(features_dir / "merged_classification.csv", index=False)
    
    cf_shift_by_class = compute_cf_shift_by_class(merged_df)
    save_json(cf_shift_by_class, metrics_dir / "cf_shift_by_class.json")
    
    visual_evidence = identify_visual_evidence_features(merged_df)
    save_json(visual_evidence, features_dir / "visual_evidence_features.json")
    
    print("Analyzing superposition...")
    superposition_results = analyze_all_compressed_only_features(
        crosscoder, classification_df, feature_activations, method
    )
    save_superposition_results(superposition_results, features_dir / "superposition_analysis.json")
    
    print("Computing aggregate metrics...")
    training_history = load_json(metrics_dir / "training_metrics.json")
    
    aggregate_metrics = compute_all_primary_metrics(
        classification_df, merged_df, superposition_results, training_history
    )
    save_metrics(aggregate_metrics, metrics_dir / "aggregate_metrics.json")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Feature classification: {features_dir / 'feature_classification.csv'}")
    print(f"  - CF scores: {features_dir / 'counterfactual_scores.csv'}")
    print(f"  - Superposition: {features_dir / 'superposition_analysis.json'}")
    print(f"  - Aggregate metrics: {metrics_dir / 'aggregate_metrics.json'}")
    
    print("\nAnalysis complete!")


def run_visualize(model: str, method: str, component: str, token_type: str):
    from .visualize import generate_all_plots
    import pandas as pd

    print(f"\n{'='*60}")
    print(f"VISUALIZATION: {model}/{method}/{component}/{token_type}")
    print(f"{'='*60}")

    results_dir = get_results_dir(model, method, component, token_type)
    features_dir = get_features_dir(results_dir)
    metrics_dir = get_metrics_dir(results_dir)
    plots_dir = get_plots_dir(results_dir)

    loss_curves_path = plots_dir / "loss_curves.png"
    if loss_curves_path.exists():
        print(f"Plots already exist at {plots_dir}, skipping visualization.")
        return

    training_history = load_json(metrics_dir / "training_metrics.json")
    classification_df = pd.read_csv(features_dir / "feature_classification.csv")
    merged_df = pd.read_csv(features_dir / "merged_classification.csv")
    superposition_results = load_json(features_dir / "superposition_analysis.json")
    
    generate_all_plots(
        training_history=training_history,
        classification_df=classification_df,
        merged_df=merged_df,
        superposition_results=superposition_results,
        plots_dir=plots_dir,
    )
    
    print(f"\nPlots saved to: {plots_dir}")
    print("Visualization complete!")


def run_all(model: str, method: str, component: str, token_type: str):
    run_extract(model, method, component, token_type)
    run_train(model, method, component, token_type)
    run_analyze(model, method, component, token_type)
    run_visualize(model, method, component, token_type)


def run_all_configurations():
    from itertools import product
    
    configurations = []
    
    for model in config.MODELS:
        for method in config.METHODS:
            for component in ["V", "P"]:
                if component == "V":
                    for token_type in config.TOKEN_TYPES:
                        configurations.append((model, method, component, token_type))
                else:
                    configurations.append((model, method, component, "cls"))
    
    for model in config.MODELS:
        for method in config.METHODS:
            configurations.append((model, method, "V_P", "cls"))
    
    print(f"\nRunning {len(configurations)} configurations:")
    for i, (model, method, component, token_type) in enumerate(configurations):
        print(f"  {i+1}. {model}/{method}/{component}/{token_type}")
    
    for i, (model, method, component, token_type) in enumerate(configurations):
        print(f"\n{'#'*60}")
        print(f"# Configuration {i+1}/{len(configurations)}")
        print(f"# {model}/{method}/{component}/{token_type}")
        print(f"{'#'*60}")
        
        run_all(model, method, component, token_type)
    
    print("\n" + "="*60)
    print("ALL CONFIGURATIONS COMPLETE!")
    print("="*60)


def run_hypothesis_tests():
    from .metrics import (
        test_hypothesis_h1,
        test_hypothesis_h2,
        test_hypothesis_h3,
        test_hypothesis_h4,
        test_hypothesis_h5,
        test_hypothesis_h6,
        test_hypothesis_h7,
        test_hypothesis_h8,
        compile_all_hypothesis_results,
    )
    import pandas as pd
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    hypothesis_results = {}
    
    for model in config.MODELS:
        wanda_v_cls_dir = get_results_dir(model, "wanda", "V", "cls")
        awq_v_cls_dir = get_results_dir(model, "awq", "V", "cls")
        
        wanda_cls = pd.read_csv(get_features_dir(wanda_v_cls_dir) / "feature_classification.csv")
        awq_cls = pd.read_csv(get_features_dir(awq_v_cls_dir) / "feature_classification.csv")
        
        h1_result = test_hypothesis_h1(wanda_cls, awq_cls)
        hypothesis_results[f"H1_{model}"] = h1_result
        
        wanda_merged = pd.read_csv(get_features_dir(wanda_v_cls_dir) / "merged_classification.csv")
        h2_result = test_hypothesis_h2(wanda_merged)
        hypothesis_results[f"H2_{model}"] = h2_result
        
        wanda_sup = load_json(get_features_dir(wanda_v_cls_dir) / "superposition_analysis.json")
        awq_sup = load_json(get_features_dir(awq_v_cls_dir) / "superposition_analysis.json")
        h3_result = test_hypothesis_h3(wanda_sup, awq_sup)
        hypothesis_results[f"H3_{model}"] = h3_result
        
        wanda_v_patch_dir = get_results_dir(model, "wanda", "V", "patch")
        wanda_patch = pd.read_csv(get_features_dir(wanda_v_patch_dir) / "feature_classification.csv")
        h4_result = test_hypothesis_h4(wanda_cls, wanda_patch)
        hypothesis_results[f"H4_{model}"] = h4_result
        
        wanda_p_dir = get_results_dir(model, "wanda", "P", "cls")
        wanda_p = pd.read_csv(get_features_dir(wanda_p_dir) / "feature_classification.csv")
        h5_result = test_hypothesis_h5(wanda_cls, wanda_p)
        hypothesis_results[f"H5_{model}"] = h5_result
        
        wanda_p_merged = pd.read_csv(get_features_dir(wanda_p_dir) / "merged_classification.csv")
        h6_result = test_hypothesis_h6(wanda_p_merged)
        hypothesis_results[f"H6_{model}"] = h6_result
        
        wanda_vp_dir = get_results_dir(model, "wanda", "V_P", "cls")
        v_metrics = load_json(get_metrics_dir(wanda_v_cls_dir) / "aggregate_metrics.json")
        p_metrics = load_json(get_metrics_dir(wanda_p_dir) / "aggregate_metrics.json")
        vp_metrics = load_json(get_metrics_dir(wanda_vp_dir) / "aggregate_metrics.json")
        h7_result = test_hypothesis_h7(v_metrics, p_metrics, vp_metrics)
        hypothesis_results[f"H7_{model}"] = h7_result
    
    blip_p_dir = get_results_dir("blip2", "wanda", "P", "cls")
    qwen3vl_p_dir = get_results_dir("qwen3vl", "wanda", "P", "cls")
    blip_p_metrics = load_json(get_metrics_dir(blip_p_dir) / "aggregate_metrics.json")
    qwen3vl_p_metrics = load_json(get_metrics_dir(qwen3vl_p_dir) / "aggregate_metrics.json")
    h8_result = test_hypothesis_h8(blip_p_metrics, qwen3vl_p_metrics)
    hypothesis_results["H8"] = h8_result
    
    results_df = compile_all_hypothesis_results(hypothesis_results)
    
    output_dir = config.CROSSCODER_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "hypothesis_results.csv", index=False)
    save_json(hypothesis_results, output_dir / "hypothesis_results.json")
    
    print("\nHypothesis Test Results:")
    print(results_df.to_string())
    print(f"\nResults saved to: {output_dir / 'hypothesis_results.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="SPARC Cross-Coder for Compressed VLM Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.crosscoder.main --model blip2 --method wanda --component V --token_type cls --stage all
  python -m src.crosscoder.main --stage all_configs
  python -m src.crosscoder.main --stage hypothesis_tests
        """
    )
    
    parser.add_argument("--model", type=str, choices=config.MODELS, help="Model name (blip2, qwen3vl, or llava15)")
    parser.add_argument("--method", type=str, choices=config.METHODS, help="Compression method (wanda or awq)")
    parser.add_argument("--component", type=str, choices=config.COMPONENTS, help="Component (V, P, or V_P)")
    parser.add_argument("--token_type", type=str, choices=config.TOKEN_TYPES, default="cls", help="Token type for vision encoder (cls or patch)")
    parser.add_argument("--stage", type=str, required=True,
                       choices=["extract", "train", "analyze", "visualize", "all", 
                               "all_configs", "hypothesis_tests"],
                       help="Stage to run")
    
    args = parser.parse_args()
    
    set_seed()
    
    if args.stage == "all_configs":
        run_all_configurations()
        return
    
    if args.stage == "hypothesis_tests":
        run_hypothesis_tests()
        return
    
    if not all([args.model, args.method, args.component]):
        parser.error("--model, --method, and --component are required for single configuration runs")
    
    if args.stage == "extract":
        run_extract(args.model, args.method, args.component, args.token_type)
    elif args.stage == "train":
        run_train(args.model, args.method, args.component, args.token_type)
    elif args.stage == "analyze":
        run_analyze(args.model, args.method, args.component, args.token_type)
    elif args.stage == "visualize":
        run_visualize(args.model, args.method, args.component, args.token_type)
    elif args.stage == "all":
        run_all(args.model, args.method, args.component, args.token_type)


if __name__ == "__main__":
    main()
