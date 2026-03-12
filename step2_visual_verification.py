"""
step2_visual_verification.py
============================
Visual verification of atomic claims using Qwen2.5-VL-7B-Instruct as judge.

For each pair:
  - Loads image1 and image2 from the image directory
  - Loads extracted claims from Step 1 JSONL (one per pair_id)
  - Asks Qwen2.5-VL to judge each claim on two dimensions:
      visual_support  (1-5): Is the claim grounded in what is visible?
      visual_accuracy (1-5): Is the claim correct given what you actually see?
  - Writes one output JSONL per pair mirroring Step 1 structure

Score rubric:
  5 = Fully supported / fully accurate
  4 = Mostly supported / mostly accurate, minor uncertainty
  3 = Partially supported / partially accurate
  2 = Weakly supported / mostly inaccurate
  1 = Not supported / clearly wrong or unverifiable

Input:
  CSV with columns: pair_id, image1, image2  (image filenames, not full paths)
  Step 1 claims dir: results/claims/{pair_id}.jsonl
  Image dir: data/images/

Output:
  results/verification/{pair_id}.jsonl
  Line 1 : metadata  {pair_id, status, n_claims, n_verified}
  Lines 2+: one verified claim per line (all Step 1 fields + judge scores)

Usage:
    python step2_visual_verification.py \
        --csv         data/pairs.csv \
        --claims_dir  results/claims \
        --image_dir   data/images \
        --output_dir  results/verification \
        --model       Qwen/Qwen2.5-VL-7B-Instruct \
        --gpus        1,2,3
"""

import os
import re
import json
import time
import random
import base64
import logging
import argparse
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp
import pandas as pd
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Qwen2.5-VL uses its own model class
from transformers import Qwen2_5_VLForConditionalGeneration


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
SYSTEM_PROMPT = """You are a rigorous visual forensics judge evaluating claims about two face images.
You will be shown Image 1 and Image 2, followed by a single atomic claim about them.
Score the claim on exactly two dimensions using integers from 1 to 5.

VISUAL_SUPPORT — Is this claim grounded in something actually visible in the images?
  5 = The feature is clearly visible and the claim directly describes it
  4 = The feature is visible, minor uncertainty due to image quality or angle
  3 = The feature is partially visible or ambiguous
  2 = The feature is barely visible or heavily obscured
  1 = The feature is not visible at all, or the claim is unverifiable from these images

VISUAL_ACCURACY — Given what you can see, is the claim correct?
  5 = Completely accurate description of what is observed
  4 = Mostly accurate, trivial deviation
  3 = Partially accurate, some aspects correct some wrong
  2 = Mostly inaccurate, description does not match observation
  1 = Clearly wrong, contradicts what is visible

Output ONLY a valid JSON object with exactly these fields. No explanation. No markdown.
{
  "visual_support": <int 1-5>,
  "visual_accuracy": <int 1-5>,
  "support_rationale": "<one sentence referencing what you see>",
  "accuracy_rationale": "<one sentence referencing what you see>"
}"""

USER_TEMPLATE = (
    "Image 1 is the first image above. Image 2 is the second image above.\n\n"
    "Claim to evaluate:\n{claim_text}\n\n"
    "Claim type: {claim_type} | Feature: {feature_category} | "
    "Images referenced: {image_reference}"
)


# ──────────────────────────────────────────────
# Image encoding
# ──────────────────────────────────────────────
def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")


# ──────────────────────────────────────────────
# Per-file output helpers (mirrors Step 1)
# ──────────────────────────────────────────────
def get_output_path(output_dir: Path, pair_id: str) -> Path:
    return output_dir / f"{pair_id}.jsonl"


def is_already_processed(output_dir: Path, pair_id: str) -> bool:
    p = get_output_path(output_dir, pair_id)
    return p.exists() and p.stat().st_size > 0


def write_verification_jsonl(output_dir: Path, pair_id: str, record: dict):
    """
    Line 1 : metadata header
    Lines 2+: one verified claim per line
    """
    out_path = get_output_path(output_dir, pair_id)
    with open(out_path, "w", encoding="utf-8") as f:
        meta = {
            "pair_id":    record["pair_id"],
            "status":     record["status"],
            "n_claims":   record["n_claims"],
            "n_verified": record["n_verified"],
            **({"error_message": record["error_message"]}
               if "error_message" in record else {}),
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for claim in record.get("verified_claims", []):
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# Load Step 1 claims for a pair
# ──────────────────────────────────────────────
def load_claims(claims_dir: Path, pair_id: str) -> list:
    """
    Read Step 1 JSONL. Skip line 1 (metadata). Return list of claim dicts.
    Only returns claims with status=success implicitly (skips meta line).
    """
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
                    # Skip metadata line
                    if obj.get("status") != "success":
                        return []  # Step 1 failed for this pair
                    continue
                claims.append(obj)
            except Exception:
                continue
    return claims


# ──────────────────────────────────────────────
# JSON extraction (robust)
# ──────────────────────────────────────────────
def extract_json_object(text: str) -> Optional[dict]:
    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 2. Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    # 3. Extract first {...} block
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────
# Single claim inference
# ──────────────────────────────────────────────
def verify_claim(model, processor, claim: dict,
                 img1_path: Path, img2_path: Path, device: str, model_name: str) -> dict:
    """Run Qwen2.5-VL on one claim + image pair. Returns judgment dict."""

    user_text = USER_TEMPLATE.format(
        claim_text=claim.get("claim_text", ""),
        claim_type=claim.get("claim_type", ""),
        feature_category=claim.get("feature_category", ""),
        image_reference=claim.get("image_reference", ""),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img1_path)},
                {"type": "image", "image": str(img2_path)},
                {"type": "text",  "text": user_text},
            ],
        },
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)

    judgment = extract_json_object(raw)
    if judgment is None:
        judgment = {
            "visual_support":      None,
            "visual_accuracy":     None,
            "support_rationale":   None,
            "accuracy_rationale":  None,
            "parse_error":         raw[:300],
        }

    # Merge Step 1 claim fields with Step 2 judgment
    return {**claim, **judgment, "judge_model": model_name}


# ──────────────────────────────────────────────
# Worker entry point
# ──────────────────────────────────────────────
def _worker_entry(gpu_id, rows, claims_dir, image_dir, output_dir, model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker(gpu_id, rows, claims_dir, image_dir, output_dir, model_name)


def worker(gpu_id, rows, claims_dir, image_dir, output_dir, model_name):
    log    = get_logger(gpu_id)
    device = "cuda:0"

    # Stagger model loading to avoid N nodes hammering the shared filesystem for the
    # same ~14 GB weights simultaneously.  Strategy: derive a deterministic base
    # offset from the SLURM job ID (so every job lands in a different 60-s bucket)
    # then add per-GPU jitter so workers within the same job also spread out.
    # slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    # job_bucket   = (slurm_job_id % 20) * 15        # 0, 15, 30, … 285 s  (20-slot ring)
    # gpu_jitter   = gpu_id * 15                     # within job: 0, 15, 30, 45 s per GPU
    # extra_jitter = random.uniform(0, 15)            # ±15 s random on top
    # wait_time    = job_bucket + gpu_jitter + extra_jitter
    # log.info(
    #     f"GPU {gpu_id} staggering {wait_time:.1f}s before model load "
    #     f"(job_bucket={job_bucket}s, gpu_jitter={gpu_jitter}s, random={extra_jitter:.1f}s)"
    # )
    # time.sleep(wait_time)

    log.info(f"Loading {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,   # bfloat16 recommended for VL models
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    log.info(f"Model loaded. Processing {len(rows)} pairs.")

    stats = {"success": 0, "skipped": 0, "error": 0, "missing_claims": 0}
    t0    = time.time()

    # Randomize the order
    random.shuffle(rows)

    for i, row in enumerate(rows):
        pair_id = str(row["pair_id"])

        # Resume check
        if is_already_processed(output_dir, pair_id):
            stats["skipped"] += 1
            continue
            
        try:
            log.info(f"Processing pair {pair_id} ({i+1}/{len(rows)}) ...")
            # Resolve image paths
            img1_path = Path(image_dir) / row["Image1"]
            img2_path = Path(image_dir) / row["Image2"]

            if not img1_path.exists() or not img2_path.exists():
                raise FileNotFoundError(
                    f"Missing images: {img1_path} | {img2_path}"
                )

            # Load Step 1 claims
            claims = load_claims(claims_dir, pair_id)
            if not claims:
                log.warning(f"No claims found for {pair_id}")
                record = {
                    "pair_id":        pair_id,
                    "status":         "missing_claims",
                    "n_claims":       0,
                    "n_verified":     0,
                    "verified_claims": [],
                }
                stats["missing_claims"] += 1
                write_verification_jsonl(output_dir, pair_id, record)
                continue

            # Verify each claim
            verified = []
            for claim in claims:
                try:
                    result = verify_claim(
                        model, processor, claim,
                        img1_path, img2_path, device, model_name
                    )
                    verified.append(result)
                except Exception as ce:
                    log.warning(f"Claim error {pair_id}/{claim.get('claim_id')}: {ce}")
                    verified.append({
                        **claim,
                        "visual_support":     None,
                        "visual_accuracy":    None,
                        "support_rationale":  None,
                        "accuracy_rationale": None,
                        "claim_error":        str(ce),
                    })

            record = {
                "pair_id":         pair_id,
                "status":          "success",
                "n_claims":        len(claims),
                "n_verified":      sum(1 for v in verified
                                       if v.get("visual_support") is not None),
                "verified_claims": verified,
            }
            stats["success"] += 1

        except Exception as e:
            log.error(f"Pair error {pair_id}: {e}\n{traceback.format_exc()}")
            record = {
                "pair_id":         pair_id,
                "status":          "error",
                "n_claims":        0,
                "n_verified":      0,
                "error_message":   str(e),
                "verified_claims": [],
            }
            stats["error"] += 1

        write_verification_jsonl(output_dir, pair_id, record)

        # Progress every 50 pairs (each pair has ~30 claims = many inferences)
        if (i + 1) % 50 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (len(rows) - i - 1) / rate
            log.info(
                f"Progress: {i+1}/{len(rows)} pairs | "
                f"rate={rate:.2f} pairs/s | "
                f"ETA={remaining/60:.1f} min | "
                f"success={stats['success']} err={stats['error']} skip={stats['skipped']}"
            )

    log.info(f"Done. Final stats: {stats}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 2: Visual Verification")
    parser.add_argument("--csv",        type=str, required=True,
                        help="CSV with columns: pair_id, image1, image2")
    parser.add_argument("--claims_dir", type=str, required=True,
                        help="Step 1 output directory (per-pair JSONL files)")
    parser.add_argument("--image_dir",  type=str, required=True,
                        help="Directory containing image files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-pair verification JSONL")
    parser.add_argument("--model",      type=str,
                        default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--gpus",       type=str, default="0",
                        help="Comma-separated GPU IDs, e.g. 1,2,3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n_gpus  = len(gpu_ids)

    # Load CSV
    df = pd.read_csv(args.csv)
    required_cols = {"pair_id", "Image1", "Image2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    rows = df.to_dict(orient="records")
    print(f"Loaded {len(rows)} pairs from CSV")

    # Resume count
    already_done = sum(
        1 for r in rows
        if is_already_processed(output_dir, str(r["pair_id"]))
    )
    print(f"Already done: {already_done} | Remaining: {len(rows) - already_done}")

    # Split rows across GPUs (round-robin)
    gpu_splits = [[] for _ in range(n_gpus)]
    for idx, row in enumerate(rows):
        gpu_splits[idx % n_gpus].append(row)

    # Spawn workers
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=_worker_entry,
            args=(
                gpu_id,
                gpu_splits[rank],
                Path(args.claims_dir),
                args.image_dir,
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
    for jsonl_path in output_dir.glob("*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            if first:
                try:
                    meta = json.loads(first)
                    total += 1
                    if meta.get("status") == "success":
                        success += 1
                    else:
                        failed += 1
                except Exception:
                    pass

    print(f"\n{'='*50}")
    print(f"VERIFICATION COMPLETE")
    print(f"  Total pairs   : {total}")
    print(f"  Success       : {success}")
    print(f"  Failed        : {failed}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()