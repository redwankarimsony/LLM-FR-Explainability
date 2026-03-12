"""
step4_aggregate.py
==================
CPU-only aggregation script. No GPU required.

Reads outputs from Steps 1, 2, and 3 for all pairs and produces a single
analysis-ready CSV with one row per pair containing:

  From Step 1 (claim extraction):
    - n_claims_total
    - n_similarity_claims
    - n_difference_claims
    - n_reasoning_claims
    - specificity_vague_ratio       (vague claims / total)
    - specificity_moderate_ratio
    - specificity_precise_ratio
    - specificity_score             (weighted: vague=1, moderate=2, precise=3 / max)
    - top_feature_categories        (comma-separated top 3 by frequency)

  From Step 2 (visual verification):
    - n_verified_claims
    - mean_visual_support           (mean of 1-5 scores across all claims)
    - mean_visual_accuracy
    - support_precise_mean          (mean support score for precise claims only)
    - support_vague_mean            (mean support score for vague claims only)
    - n_parse_errors                (claims where judge failed to parse)

  From Step 3 (consistency check):
    - n_contradiction_pairs         (cross-claim pairs checked)
    - n_contradictions_found
    - contradiction_rate            (n_contradictions / n_pairs checked)
    - verdict_alignment             (ENTAILMENT / NEUTRAL / CONTRADICTION)
    - verdict_alignment_confidence
    - generated_verdict             (Match / Non-match / Uncertain)

  From CSV (ground truth):
    - label                         (0 / 1)
    - verdict_correct               (1 if generated_verdict matches label, else 0)

Usage:
    python step4_aggregate.py \
        --csv              data/pairs.csv \
        --claims_dir       results/claims \
        --verification_dir results/verification \
        --consistency_dir  results/consistency \
        --explanations_dir data/explanations \
        --output           results/summary.csv
"""

import re
import json
import argparse
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

SPECIFICITY_WEIGHT = {"vague": 1, "moderate": 2, "precise": 3}


# ──────────────────────────────────────────────
# JSONL readers
# ──────────────────────────────────────────────
def read_jsonl(path: Path) -> list:
    """Read all lines of a JSONL file. Returns (meta, claims_list)."""
    if not path.exists():
        return None, []
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    if not lines:
        return None, []
    return lines[0], lines[1:]  # (metadata, rest)


# ──────────────────────────────────────────────
# Step 1 aggregation
# ──────────────────────────────────────────────
def aggregate_step1(claims_dir: Path, pair_id: str) -> dict:
    meta, claims = read_jsonl(claims_dir / f"{pair_id}.jsonl")
    base = {
        "s1_status":                "missing",
        "n_claims_total":           0,
        "n_similarity_claims":      0,
        "n_difference_claims":      0,
        "n_reasoning_claims":       0,
        "specificity_vague_ratio":  None,
        "specificity_moderate_ratio": None,
        "specificity_precise_ratio":  None,
        "specificity_score":        None,
        "top_feature_categories":   None,
    }
    if meta is None or meta.get("status") != "success" or not claims:
        return base

    base["s1_status"]       = "success"
    base["n_claims_total"]  = len(claims)

    # Claim type counts
    type_counts = {"similarity": 0, "difference": 0, "reasoning": 0}
    for c in claims:
        t = c.get("claim_type", "")
        if t in type_counts:
            type_counts[t] += 1
    base["n_similarity_claims"] = type_counts["similarity"]
    base["n_difference_claims"] = type_counts["difference"]
    base["n_reasoning_claims"]  = type_counts["reasoning"]

    # Specificity
    spec_counts = {"vague": 0, "moderate": 0, "precise": 0}
    for c in claims:
        s = c.get("specificity", "")
        if s in spec_counts:
            spec_counts[s] += 1
    total = sum(spec_counts.values())
    if total > 0:
        base["specificity_vague_ratio"]    = round(spec_counts["vague"]    / total, 4)
        base["specificity_moderate_ratio"] = round(spec_counts["moderate"] / total, 4)
        base["specificity_precise_ratio"]  = round(spec_counts["precise"]  / total, 4)
        weighted = sum(SPECIFICITY_WEIGHT[k] * v for k, v in spec_counts.items())
        base["specificity_score"] = round(weighted / (total * 3), 4)  # normalised 0-1

    # Top feature categories
    feat_counts = {}
    for c in claims:
        f = c.get("feature_category", "unknown")
        feat_counts[f] = feat_counts.get(f, 0) + 1
    top3 = sorted(feat_counts, key=feat_counts.get, reverse=True)[:3]
    base["top_feature_categories"] = ",".join(top3)

    return base


# ──────────────────────────────────────────────
# Step 2 aggregation
# ──────────────────────────────────────────────
def aggregate_step2(verification_dir: Path, pair_id: str) -> dict:
    meta, claims = read_jsonl(verification_dir / f"{pair_id}.jsonl")
    base = {
        "s2_status":           "missing",
        "n_verified_claims":   0,
        "mean_visual_support": None,
        "mean_visual_accuracy": None,
        "support_precise_mean": None,
        "support_vague_mean":   None,
        "n_parse_errors":       0,
    }
    if meta is None or meta.get("status") != "success" or not claims:
        return base

    base["s2_status"]         = "success"
    base["n_verified_claims"] = meta.get("n_verified", 0)

    support_scores  = []
    accuracy_scores = []
    precise_support = []
    vague_support   = []
    parse_errors    = 0

    for c in claims:
        vs = c.get("visual_support")
        va = c.get("visual_accuracy")
        if vs is None and va is None:
            parse_errors += 1
            continue
        if vs is not None:
            support_scores.append(vs)
            spec = c.get("specificity", "")
            if spec == "precise":
                precise_support.append(vs)
            elif spec == "vague":
                vague_support.append(vs)
        if va is not None:
            accuracy_scores.append(va)

    base["n_parse_errors"] = parse_errors
    if support_scores:
        base["mean_visual_support"]  = round(sum(support_scores)  / len(support_scores),  4)
    if accuracy_scores:
        base["mean_visual_accuracy"] = round(sum(accuracy_scores) / len(accuracy_scores), 4)
    if precise_support:
        base["support_precise_mean"] = round(sum(precise_support) / len(precise_support), 4)
    if vague_support:
        base["support_vague_mean"]   = round(sum(vague_support)   / len(vague_support),   4)

    return base


# ──────────────────────────────────────────────
# Step 3 aggregation
# ──────────────────────────────────────────────
def aggregate_step3(consistency_dir: Path, pair_id: str) -> dict:
    meta, results = read_jsonl(consistency_dir / f"{pair_id}.jsonl")
    base = {
        "s3_status":                  "missing",
        "n_contradiction_pairs":      0,
        "n_contradictions_found":     0,
        "contradiction_rate":         None,
        "verdict_alignment":          None,
        "verdict_alignment_confidence": None,
        "generated_verdict":          None,
    }
    if meta is None or meta.get("status") != "success":
        return base

    base["s3_status"]              = "success"
    base["n_contradiction_pairs"]  = meta.get("n_contradiction_pairs", 0)
    base["n_contradictions_found"] = meta.get("n_contradictions_found", 0)
    base["verdict_alignment"]      = meta.get("verdict_alignment")

    n_pairs = base["n_contradiction_pairs"]
    n_found = base["n_contradictions_found"]
    if n_pairs > 0:
        base["contradiction_rate"] = round(n_found / n_pairs, 4)

    # Extract verdict alignment confidence and generated verdict from last line
    for r in results:
        if r.get("check_type") == "verdict_alignment":
            base["verdict_alignment_confidence"] = r.get("confidence")
            base["generated_verdict"]            = r.get("verdict")
            break

    return base


# ──────────────────────────────────────────────
# Verdict correctness
# ──────────────────────────────────────────────
def compute_verdict_correct(generated_verdict: str, label) -> int:
    """
    Compare model's generated verdict against ground truth label.
      label 1 = Match
      label 0 = Non-match
    Returns 1 if correct, 0 if wrong, None if Uncertain or missing.
    """
    if generated_verdict is None or pd.isna(label):
        return None
    if generated_verdict == "Uncertain":
        return None  # Cannot assess correctness for Uncertain
    gt = "Match" if int(label) == 1 else "Non-match"
    return 1 if generated_verdict == gt else 0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 4: Aggregate all steps into summary CSV")
    parser.add_argument("--csv",              type=str, required=True,
                        help="Original CSV with pair_id and label columns")
    parser.add_argument("--claims_dir",       type=str, required=True,
                        help="Step 1 output directory")
    parser.add_argument("--verification_dir", type=str, required=True,
                        help="Step 2 output directory")
    parser.add_argument("--consistency_dir",  type=str, required=True,
                        help="Step 3 output directory")
    parser.add_argument("--output",           type=str, required=True,
                        help="Output CSV path e.g. results/summary.csv")
    args = parser.parse_args()

    claims_dir       = Path(args.claims_dir)
    verification_dir = Path(args.verification_dir)
    consistency_dir  = Path(args.consistency_dir)
    output_path      = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original CSV
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} pairs from CSV")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating"):
        pair_id = str(row["pair_id"])
        label   = row.get("Label", None)

        s1 = aggregate_step1(claims_dir,       pair_id)
        s2 = aggregate_step2(verification_dir, pair_id)
        s3 = aggregate_step3(consistency_dir,  pair_id)

        # Verdict correctness
        verdict_correct = None
        if label is not None and s3.get("generated_verdict") is not None:
            verdict_correct = compute_verdict_correct(
                s3["generated_verdict"], label
            )

        combined = {
            "pair_id":         pair_id,
            "Label":           label,
            "verdict_correct": verdict_correct,
            **s1,
            **s2,
            **s3,
        }
        rows.append(combined)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)

    # ── Print summary statistics ──
    success_mask = summary_df["s1_status"] == "success"
    print(f"\n{'='*60}")
    print(f"AGGREGATION COMPLETE")
    print(f"  Total pairs              : {len(summary_df)}")
    print(f"  Step 1 success           : {(summary_df['s1_status'] == 'success').sum()}")
    print(f"  Step 2 success           : {(summary_df['s2_status'] == 'success').sum()}")
    print(f"  Step 3 success           : {(summary_df['s3_status'] == 'success').sum()}")

    if success_mask.any():
        print(f"\n  --- Specificity (Step 1) ---")
        print(f"  Mean specificity score   : {summary_df.loc[success_mask, 'specificity_score'].mean():.4f}")
        print(f"  Mean precise ratio       : {summary_df.loc[success_mask, 'specificity_precise_ratio'].mean():.4f}")
        print(f"  Mean vague ratio         : {summary_df.loc[success_mask, 'specificity_vague_ratio'].mean():.4f}")

        s2_mask = summary_df["s2_status"] == "success"
        if s2_mask.any():
            print(f"\n  --- Visual Verification (Step 2) ---")
            print(f"  Mean visual support      : {summary_df.loc[s2_mask, 'mean_visual_support'].mean():.4f}")
            print(f"  Mean visual accuracy     : {summary_df.loc[s2_mask, 'mean_visual_accuracy'].mean():.4f}")
            print(f"  Support precise vs vague : {summary_df.loc[s2_mask, 'support_precise_mean'].mean():.4f} vs {summary_df.loc[s2_mask, 'support_vague_mean'].mean():.4f}")

        s3_mask = summary_df["s3_status"] == "success"
        if s3_mask.any():
            print(f"\n  --- Consistency (Step 3) ---")
            print(f"  Mean contradiction rate  : {summary_df.loc[s3_mask, 'contradiction_rate'].mean():.4f}")
            print(f"  Verdict alignment dist.  :")
            print(summary_df.loc[s3_mask, "verdict_alignment"].value_counts().to_string(header=False))
            print(f"\n  Generated verdict dist.  :")
            print(summary_df.loc[s3_mask, "generated_verdict"].value_counts().to_string(header=False))

        if "verdict_correct" in summary_df and summary_df["verdict_correct"].notna().any():
            acc = summary_df["verdict_correct"].mean()
            print(f"\n  --- Verdict Accuracy ---")
            print(f"  Verdict accuracy         : {acc:.4f}  ({acc*100:.1f}%)")

    print(f"\n  Output saved to          : {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()