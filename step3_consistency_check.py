"""
step3_consistency_check.py
==========================
Consistency checking of extracted claims using Qwen2.5-14B-Instruct as judge.

Two sub-checks per pair:

  3a. Cross-claim contradiction
      For every pair of claims sharing the same feature_category where one is
      a "similarity" and the other is a "difference", ask Qwen whether they
      contradict each other.
      Labels: CONTRADICTION | NEUTRAL | CONSISTENT

  3b. Reasoning-verdict alignment
      Collect all "overall_reasoning" claims into a reasoning paragraph.
      Extract the stated verdict (Match / Non-match) from the metadata.
      Ask Qwen whether the reasoning entails, is neutral toward, or contradicts
      the verdict.
      Labels: ENTAILMENT | NEUTRAL | CONTRADICTION

Input:
  Step 1 claims dir    : results/claims/{pair_id}.jsonl
  Explanations dir     : data/explanations/{pair_id}.txt  (original model output)
  CSV                  : data/pairs.csv  (pair_id column)

Output:
  results/consistency/{pair_id}.jsonl
  Line 1   : metadata  {pair_id, status, n_contradiction_pairs, verdict_alignment}
  Lines 2+ : one contradiction-check result per claim pair
  Last line: verdict alignment result

Usage:
    python step3_consistency_check.py \
        --csv              data/pairs.csv \
        --claims_dir       results/claims \
        --explanations_dir data/explanations \
        --output_dir       results/consistency \
        --model            Qwen/Qwen2.5-14B-Instruct \
        --gpus             1,2,3
"""

import os
import re
import json
import time
import logging
import argparse
import traceback
import itertools
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [GPU %(gpu)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(gpu_id: int):
    logger = logging.getLogger(f"gpu{gpu_id}")
    return logging.LoggerAdapter(logger, {"gpu": gpu_id})


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────
CONTRADICTION_SYSTEM = """You are a logical consistency judge for face comparison claims.
You will be given two atomic claims about the same pair of face images.
Claim A is a similarity claim. Claim B is a difference claim.
Both are about the same facial feature category.

Determine whether the two claims are logically consistent or contradictory.

CONTRADICTION : The claims directly conflict — one says the feature is the same,
                the other says it differs, and both cannot be true simultaneously.
NEUTRAL       : The claims discuss different aspects of the same feature category
                and do not directly conflict.
CONSISTENT    : The claims are complementary and reinforce each other.

Output ONLY a valid JSON object. No explanation. No markdown.
{
  "label": "CONTRADICTION" | "NEUTRAL" | "CONSISTENT",
  "confidence": <int 1-5>,
  "rationale": "<one sentence explaining your judgment>"
}"""

CONTRADICTION_USER = """Feature category: {feature_category}

Claim A (similarity): {claim_a}

Claim B (difference): {claim_b}"""


VERDICT_ALIGNMENT_SYSTEM = """You are a logical consistency judge for face comparison reasoning.
You will be given a reasoning paragraph and a final verdict about whether two face
images show the same person. The verdict is one of: Match, Non-match, or Uncertain.

Determine whether the reasoning supports, is neutral toward, or contradicts the verdict.

ENTAILMENT    : The reasoning logically supports and leads to the stated verdict.
                e.g. Strong similarities described → Verdict: Match
                e.g. Major differences described → Verdict: Non-match
                e.g. Mixed inconclusive evidence → Verdict: Uncertain
NEUTRAL       : The reasoning is present but does not clearly support or oppose
                the verdict.
CONTRADICTION : The reasoning contradicts the stated verdict — the evidence described
                would lead to the opposite conclusion.
                e.g. Strong similarities described → Verdict: Non-match
                e.g. Clear differences described → Verdict: Match
                e.g. Definitive conclusion in reasoning → Verdict: Uncertain

Output ONLY a valid JSON object. No explanation. No markdown.
{
  "label": "ENTAILMENT" | "NEUTRAL" | "CONTRADICTION",
  "confidence": <int 1-5>,
  "rationale": "<one sentence explaining your judgment>"
}"""

VERDICT_ALIGNMENT_USER = """Verdict: {verdict}

Reasoning:
{reasoning}"""


# ──────────────────────────────────────────────
# Per-file output helpers (mirrors Step 1 & 2)
# ──────────────────────────────────────────────
def get_output_path(output_dir: Path, pair_id: str) -> Path:
    return output_dir / f"{pair_id}.jsonl"


def is_already_processed(output_dir: Path, pair_id: str) -> bool:
    p = get_output_path(output_dir, pair_id)
    return p.exists() and p.stat().st_size > 0


def write_consistency_jsonl(output_dir: Path, pair_id: str, record: dict):
    """
    Line 1      : metadata header
    Lines 2..N  : one cross-claim contradiction result per line
    Last line   : verdict alignment result
    """
    out_path = get_output_path(output_dir, pair_id)
    with open(out_path, "w", encoding="utf-8") as f:
        # Metadata
        meta = {
            "pair_id":                 record["pair_id"],
            "status":                  record["status"],
            "n_contradiction_pairs":   record.get("n_contradiction_pairs", 0),
            "n_contradictions_found":  record.get("n_contradictions_found", 0),
            "verdict_alignment":       record.get("verdict_alignment", None),
            **({"error_message": record["error_message"]}
               if "error_message" in record else {}),
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # Cross-claim contradiction results
        for result in record.get("contradiction_results", []):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Verdict alignment result
        if record.get("verdict_result"):
            f.write(json.dumps(record["verdict_result"], ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# Load Step 1 claims
# ──────────────────────────────────────────────
def load_claims(claims_dir: Path, pair_id: str) -> list:
    claims_path = claims_dir / f"{pair_id}.jsonl"
    if not claims_path.exists():
        return []
    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if i == 0:
                    if obj.get("status") != "success":
                        return []
                    continue
                claims.append(obj)
            except Exception:
                continue
    return claims


# ──────────────────────────────────────────────
# Parse generated verdict from explanation .txt
# ──────────────────────────────────────────────
def parse_generated_verdict(explanations_dir: Path, pair_id: str) -> Optional[str]:
    """
    Parse the model-generated verdict from the original .txt explanation file.
    Handles three possible verdicts:
      'Match Verdict: Match'
      'Match Verdict: Non-match'
      'Match Verdict: Uncertain'
    Returns 'Match', 'Non-match', 'Uncertain', or None if not found.
    """
    txt_path = explanations_dir / f"{pair_id}.txt"
    if not txt_path.exists():
        return None

    text = txt_path.read_text(encoding="utf-8")
    pattern = re.search(
        r"(?:match\s+)?verdict\s*:\s*(match|non-match|nonmatch|no\s+match|uncertain)",
        text,
        re.IGNORECASE,
    )
    if pattern:
        raw = pattern.group(1).strip().lower()
        if raw == "match":
            return "Match"
        elif raw in ("uncertain",):
            return "Uncertain"
        else:
            return "Non-match"
    return None


# ──────────────────────────────────────────────
# JSON extraction
# ──────────────────────────────────────────────
def extract_json_object(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────
def run_inference(model, tokenizer, system: str, user: str, device: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ──────────────────────────────────────────────
# 3a. Cross-claim contradiction check
# ──────────────────────────────────────────────
def check_contradictions(model, tokenizer, claims: list, device: str) -> list:
    """
    For each feature_category that has both similarity and difference claims,
    run pairwise contradiction checks.
    Returns list of result dicts.
    """
    # Group by feature_category
    by_feature: dict = {}
    for c in claims:
        cat = c.get("feature_category", "unknown")
        by_feature.setdefault(cat, {"similarity": [], "difference": []})
        ctype = c.get("claim_type", "")
        if ctype == "similarity":
            by_feature[cat]["similarity"].append(c)
        elif ctype == "difference":
            by_feature[cat]["difference"].append(c)

    results = []
    for feature, groups in by_feature.items():
        sim_claims  = groups["similarity"]
        diff_claims = groups["difference"]

        if not sim_claims or not diff_claims:
            continue  # No cross-type pair possible for this feature

        # Check all (similarity, difference) pairs for this feature
        for sim_c, diff_c in itertools.product(sim_claims, diff_claims):
            user_text = CONTRADICTION_USER.format(
                feature_category=feature,
                claim_a=sim_c.get("claim_text", ""),
                claim_b=diff_c.get("claim_text", ""),
            )
            raw = run_inference(
                model, tokenizer, CONTRADICTION_SYSTEM, user_text, device
            )
            judgment = extract_json_object(raw)
            if judgment is None:
                judgment = {
                    "label":      "PARSE_ERROR",
                    "confidence": None,
                    "rationale":  None,
                    "raw":        raw[:200],
                }

            results.append({
                "check_type":       "cross_claim_contradiction",
                "feature_category": feature,
                "claim_a_id":       sim_c.get("claim_id"),
                "claim_a_text":     sim_c.get("claim_text"),
                "claim_b_id":       diff_c.get("claim_id"),
                "claim_b_text":     diff_c.get("claim_text"),
                **judgment,
            })

    return results


# ──────────────────────────────────────────────
# 3b. Reasoning-verdict alignment
# ──────────────────────────────────────────────
def check_verdict_alignment(
    model, tokenizer, claims: list, verdict: str, device: str
) -> dict:
    """
    Collect overall_reasoning claims into a paragraph.
    Check whether the reasoning entails the verdict.
    """
    reasoning_claims = [
        c.get("claim_text", "")
        for c in claims
        if c.get("claim_type") == "reasoning"
        or c.get("feature_category") == "overall_reasoning"
    ]

    if not reasoning_claims:
        return {
            "check_type": "verdict_alignment",
            "verdict":    verdict,
            "reasoning":  "",
            "label":      "NO_REASONING_CLAIMS",
            "confidence": None,
            "rationale":  None,
        }

    reasoning_paragraph = " ".join(reasoning_claims)
    user_text = VERDICT_ALIGNMENT_USER.format(
        verdict=verdict,
        reasoning=reasoning_paragraph,
    )
    raw = run_inference(
        model, tokenizer, VERDICT_ALIGNMENT_SYSTEM, user_text, device
    )
    judgment = extract_json_object(raw)
    if judgment is None:
        judgment = {
            "label":      "PARSE_ERROR",
            "confidence": None,
            "rationale":  None,
            "raw":        raw[:200],
        }

    return {
        "check_type": "verdict_alignment",
        "verdict":    verdict,
        "reasoning":  reasoning_paragraph,
        **judgment,
    }


# ──────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────
def _worker_entry(gpu_id, rows, claims_dir, explanations_dir, output_dir, model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker(gpu_id, rows, claims_dir, explanations_dir, output_dir, model_name)


def worker(gpu_id, rows, claims_dir, explanations_dir, output_dir, model_name):
    global args_explanations_dir
    args_explanations_dir = explanations_dir
    log    = get_logger(gpu_id)
    device = "cuda:0"

    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    log.info(f"Model loaded. Processing {len(rows)} pairs.")

    stats = {"success": 0, "skipped": 0, "error": 0, "missing_claims": 0}
    t0    = time.time()

    for i, row in enumerate(rows):
        pair_id = str(row["pair_id"])

        if is_already_processed(output_dir, pair_id):
            stats["skipped"] += 1
            continue

        try:
            # Load Step 1 claims
            claims = load_claims(claims_dir, pair_id)
            if not claims:
                log.warning(f"No claims for {pair_id}")
                record = {
                    "pair_id":               pair_id,
                    "status":                "missing_claims",
                    "n_contradiction_pairs": 0,
                    "n_contradictions_found": 0,
                    "verdict_alignment":     None,
                    "contradiction_results": [],
                    "verdict_result":        None,
                }
                stats["missing_claims"] += 1
                write_consistency_jsonl(output_dir, pair_id, record)
                continue

            # ── 3a: Cross-claim contradiction ──
            contradiction_results = check_contradictions(
                model, tokenizer, claims, device
            )
            n_contradictions = sum(
                1 for r in contradiction_results
                if r.get("label") == "CONTRADICTION"
            )

            # ── 3b: Verdict alignment ──
            # Parse the model's own generated verdict from the .txt file
            verdict = parse_generated_verdict(Path(args_explanations_dir), pair_id)
            if verdict is None:
                log.warning(f"Could not parse generated verdict for {pair_id}, skipping alignment")
            verdict_result = check_verdict_alignment(
                model, tokenizer, claims, verdict or "Unknown", device
            )

            verdict_result = check_verdict_alignment(
                model, tokenizer, claims, verdict, device
            )

            record = {
                "pair_id":                pair_id,
                "status":                 "success",
                "n_contradiction_pairs":  len(contradiction_results),
                "n_contradictions_found": n_contradictions,
                "verdict_alignment":      verdict_result.get("label"),
                "contradiction_results":  contradiction_results,
                "verdict_result":         verdict_result,
            }
            stats["success"] += 1

        except Exception as e:
            log.error(f"Pair error {pair_id}: {e}\n{traceback.format_exc()}")
            record = {
                "pair_id":                pair_id,
                "status":                 "error",
                "n_contradiction_pairs":  0,
                "n_contradictions_found": 0,
                "verdict_alignment":      None,
                "error_message":          str(e),
                "contradiction_results":  [],
                "verdict_result":         None,
            }
            stats["error"] += 1

        write_consistency_jsonl(output_dir, pair_id, record)

        if (i + 1) % 100 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (len(rows) - i - 1) / rate
            log.info(
                f"Progress: {i+1}/{len(rows)} | "
                f"rate={rate:.2f} pairs/s | "
                f"ETA={remaining/60:.1f} min | "
                f"success={stats['success']} err={stats['error']}"
            )

    log.info(f"Done. Final stats: {stats}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 3: Consistency Check")
    parser.add_argument("--csv",              type=str, required=True,
                        help="CSV with pair_id column")
    parser.add_argument("--claims_dir",       type=str, required=True,
                        help="Step 1 output directory")
    parser.add_argument("--explanations_dir", type=str, required=True,
                        help="Directory containing original .txt explanation files")
    parser.add_argument("--output_dir",       type=str, required=True,
                        help="Output directory for per-pair consistency JSONL")
    parser.add_argument("--model",            type=str,
                        default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--gpus",             type=str, default="0",
                        help="Comma-separated GPU IDs e.g. 1,2,3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n_gpus  = len(gpu_ids)

    df   = pd.read_csv(args.csv)
    rows = df.to_dict(orient="records")
    print(f"Loaded {len(rows)} pairs from CSV")

    already_done = sum(
        1 for r in rows
        if is_already_processed(output_dir, str(r["pair_id"]))
    )
    print(f"Already done: {already_done} | Remaining: {len(rows) - already_done}")

    # Round-robin split
    gpu_splits = [[] for _ in range(n_gpus)]
    for idx, row in enumerate(rows):
        gpu_splits[idx % n_gpus].append(row)

    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=_worker_entry,
            args=(
                gpu_id,
                gpu_splits[rank],
                Path(args.claims_dir),
                Path(args.explanations_dir),
                output_dir,
                args.model,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Final summary
    total, success, failed = 0, 0, 0
    total_contradictions, total_misaligned = 0, 0
    for jsonl_path in output_dir.glob("*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            if first:
                try:
                    meta = json.loads(first)
                    total += 1
                    if meta.get("status") == "success":
                        success += 1
                        total_contradictions += meta.get("n_contradictions_found", 0)
                        if meta.get("verdict_alignment") == "CONTRADICTION":
                            total_misaligned += 1
                    else:
                        failed += 1
                except Exception:
                    pass

    print(f"\n{'='*50}")
    print(f"CONSISTENCY CHECK COMPLETE")
    print(f"  Total pairs            : {total}")
    print(f"  Success                : {success}")
    print(f"  Failed                 : {failed}")
    print(f"  Cross-claim contradictions found : {total_contradictions}")
    print(f"  Verdict-reasoning misaligned     : {total_misaligned}")
    print(f"  Output dir             : {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()