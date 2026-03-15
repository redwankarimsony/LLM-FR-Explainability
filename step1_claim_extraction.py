"""
step1_claim_extraction.py
=========================
Atomic claim extraction from face comparison explanation texts.
- Multi-GPU via torch.multiprocessing (one process per GPU)
- Resume-safe: skips files whose output JSONL already exists in output_dir
- Output: one JSONL file per input .txt file, mirroring the input structure

  Input:   data/explanations/explanation_00001.txt
  Output:  results/claims/explanation_00001.jsonl
           (each line = one atomic claim as a JSON object)

Usage:
    python step1_claim_extraction.py \
        --input_dir  data/explanations \
        --output_dir results/claims \
        --model      Qwen/Qwen2.5-14B-Instruct \
        --gpus       0,1,2,3
"""

import os
import re
import sys
import json
import time
import random
import logging
import argparse
import traceback
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import pandas as pd

import torch
import torch.multiprocessing as mp
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
# Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise claim extraction system for face comparison text analysis.
Given a face matching explanation, extract every atomic claim.
Break compound sentences into individual claims — one verifiable fact per claim.
Output ONLY a valid JSON array. No explanation. No markdown fences.

For each claim output a JSON object with exactly these fields:
- claim_id:         sequential string starting at C01 (C01, C02, C03 ...)
- claim_text:       the atomic claim rewritten as a complete standalone declarative sentence
- claim_type:       one of ["similarity", "difference", "reasoning"]
- feature_category: one of ["facial_structure", "eyes", "nose", "mouth_lips",
                    "eyebrows", "skin_texture", "periocular", "ears",
                    "hair_hairline", "lighting_background", "expression",
                    "pose", "overall_reasoning"]
- image_reference:  one of ["both", "image_1", "image_2"]
- specificity:      one of ["vague", "moderate", "precise"]
                    vague    = generic, no measurable detail
                    moderate = some detail but not anatomically specific
                    precise  = anatomically grounded or measurable

Rules:
1. One fact per claim — split compound sentences ruthlessly.
2. Rewrite every claim as a self-contained sentence (no dangling pronouns).
3. Do NOT infer — extract only what is explicitly stated.
4. Output ONLY a valid JSON array. No explanation. No markdown fences."""

USER_TEMPLATE = "Extract all atomic claims from the following face matching explanation:\n\n{text}"


# ──────────────────────────────────────────────
# Per-file output helpers
# ──────────────────────────────────────────────
def get_output_path(output_dir: Path, file_id: str) -> Path:
    """Return the per-file JSONL path for a given file_id."""
    return output_dir / f"{file_id}.jsonl"


def is_already_processed(output_dir: Path, file_id: str) -> bool:
    """Resume check: output JSONL exists and is non-empty."""
    p = get_output_path(output_dir, file_id)
    return p.exists() and p.stat().st_size > 0


def write_claims_jsonl(output_dir: Path, file_id: str, record: dict):
    """
    Write one JSONL file per input explanation.
    Each line = one atomic claim JSON object (flat, not nested).
    Meta fields (file_id, status, n_claims) are written as the first line.
    """
    out_path = get_output_path(output_dir, file_id)
    with open(out_path, "w", encoding="utf-8") as f:
        # Line 1: metadata header
        meta = {
            "file_id":  record["file_id"],
            "status":   record["status"],
            "n_claims": record["n_claims"],
            "image1": record["image1"],
            "image2": record["image2"],
            "label": record["label"],
            **({"raw_output": record["raw_output"]} if "raw_output" in record else {}),
            **({"error_message": record["error_message"]} if "error_message" in record else {}),
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        # Remaining lines: one claim per line
        for claim in record.get("claims", []):
            claim_with_id = {"file_id": file_id, **claim}
            f.write(json.dumps(claim_with_id, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# JSON extraction (robust)
# ──────────────────────────────────────────────
def extract_json_array(text: str) -> Optional[list]:
    """
    Try multiple strategies to extract a JSON array from model output.
    Returns parsed list or None on failure.
    """
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

    # 3. Extract first [...] block
    match = re.search(r"\[.*\]", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None


# ──────────────────────────────────────────────
# Single-file inference
# ──────────────────────────────────────────────
def run_inference(model, tokenizer, text: str, device: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ──────────────────────────────────────────────
# Worker entry point (one per GPU)
# Sets CUDA_VISIBLE_DEVICES BEFORE any CUDA context is touched
# ──────────────────────────────────────────────
def _worker_entry(gpu_id, file_paths, output_dir, model_name, batch_size, df):
    """Thin wrapper: pins CUDA_VISIBLE_DEVICES before worker() touches any CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker(gpu_id, file_paths, output_dir, model_name, batch_size, df)


# ──────────────────────────────────────────────
# Worker process (one per GPU)
# ──────────────────────────────────────────────
def worker(
    gpu_id: int,
    file_paths: list,           # subset assigned to this GPU
    output_dir: Path,
    model_name: str,
    batch_size: int,
    df: pd.DataFrame,
):
    log = get_logger(gpu_id)
    # CUDA_VISIBLE_DEVICES must be set before any CUDA context is created.
    # It was already set in the environment before this process was spawned
    # (see main()), so cuda:0 is the only visible device here.
    device = "cuda:0"

    log.info(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    log.info(f"Model loaded. Processing {len(file_paths)} files.")

    stats = {"success": 0, "json_fail": 0, "error": 0, "skipped": 0}
    t0 = time.time()

    

    # Randomize file order for better load balancing across GPUs (optional)
    random.shuffle(file_paths)

    # Convert the pair_id column of the dataframe to the index for faster lookup
    if df is not None and "pair_id" in df.columns:
        df.set_index("pair_id", inplace=True)



    # Remove the already rocessed files from the file_paths list to avoid redundant processing
    file_paths = [fp for fp in file_paths if not is_already_processed(output_dir, fp.stem)]
    pbar = tqdm(
        file_paths,
        desc=f"GPU {gpu_id}",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    

    for i, fp in enumerate(pbar):
        file_id = fp.stem  # e.g. "explanation_00042"

        # Resume check — skip if output JSONL already exists
        if is_already_processed(output_dir, file_id):
            stats["skipped"] += 1
            tqdm.write(f"[GPU {gpu_id}] SKIP  {file_id}", file=sys.stdout)
            continue

        try:
            text = fp.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError("Empty file")
            image1 = df.loc[file_id, "image1"] if df is not None and file_id in df.index else "unknown"
            image2 = df.loc[file_id, "image2"] if df is not None and file_id in df.index else "unknown"
            label = str(df.loc[file_id, "label"]) if df is not None and file_id in df.index else "unknown"

            raw_output = run_inference(model, tokenizer, text, device)
            claims = extract_json_array(raw_output)

            if claims is None:
                log.warning(f"JSON parse failed for {file_id}")
                record = {
                    "file_id":    file_id,
                    "status":     "json_parse_error",
                    "claims":     [],
                    # "raw_output": raw_output[:500],
                    "raw_output": raw_output,
                    "n_claims":   0,
                    "image1": image1,
                    "image2": image2,
                    "label": label,
                }
                stats["json_fail"] += 1
            else:
                record = {
                    "file_id":  file_id,
                    "status":   "success",
                    "claims":   claims,
                    "n_claims": len(claims),
                    "image1": image1,
                    "image2": image2,
                    "label": label,
                }
                stats["success"] += 1

        except Exception as e:
            log.error(f"Error on {file_id}: {e}\n{traceback.format_exc()}")
            record = {
                "file_id":       file_id,
                "status":        "error",
                "claims":        [],
                "error_message": str(e),
                "n_claims":      0,
                "image1": "unknown",
                "image2": "unknown",
                "label": "unknown",
            }
            stats["error"] += 1

        # Write per-file JSONL — no lock needed (each file is independent)
        write_claims_jsonl(output_dir, file_id, record)

        tqdm.write(
            f"[GPU {gpu_id}] DONE  {file_id} | "
            f"status={record['status']} | n_claims={record.get('n_claims', 0)}",
            file=sys.stdout,
        )

    log.info(f"Done. Final stats: {stats}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 1: Claim Extraction at Scale")
    parser.add_argument("--input_dir",  type=str, required=True, help="Directory of .txt explanation files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for per-file JSONL files")
    parser.add_argument("--model",      type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--gpus",       type=str, default="0",   help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=4,     help="(Reserved for future batching)")
    parser.add_argument("--glob",       type=str, default="*.txt", help="File glob pattern")
    parser.add_argument("--dataframe_path", type=str, help="Path to save the combined claims dataframe (CSV)")  
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.dataframe_path)
        print(f"Loaded dataframe with {len(df)} rows from {args.dataframe_path}")
    except Exception as e:
        print(f"Error loading dataframe from {args.dataframe_path}: {e}")
        df = None
    
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n_gpus  = len(gpu_ids)

    # Collect all input files
    all_files = sorted(input_dir.glob(args.glob))
    if not all_files:
        print(f"No files found in {input_dir} matching '{args.glob}'")
        return

    # Resume: count already-done files
    already_done = sum(1 for fp in all_files if is_already_processed(output_dir, fp.stem))
    print(f"Found {len(all_files)} files | Already done: {already_done} | Remaining: {len(all_files) - already_done}")

    # Split files across GPUs (round-robin for balanced load)
    gpu_file_splits = [[] for _ in range(n_gpus)]
    for idx, fp in enumerate(all_files):
        gpu_file_splits[idx % n_gpus].append(fp)

    # Spawn one process per GPU.
    # CUDA_VISIBLE_DEVICES is set as the very first thing inside _worker_entry,
    # before any CUDA context is created.
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=_worker_entry,
            args=(gpu_id, gpu_file_splits[rank], output_dir, args.model, args.batch_size, df),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ── Final summary ──
    total    = len(list(output_dir.glob("*.jsonl")))
    success  = 0
    failed   = 0
    for jsonl_path in output_dir.glob("*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    meta = json.loads(first_line)
                    if meta.get("status") == "success":
                        success += 1
                    else:
                        failed += 1
                except Exception:
                    pass

    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Total JSONL files : {total}")
    print(f"  Success           : {success}")
    print(f"  Failed            : {failed}")
    print(f"  Output directory  : {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()