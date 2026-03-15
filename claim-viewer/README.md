# FR Explainability Viewer

A dark-themed React application for browsing and analyzing LLM-generated facial recognition explainability claims. Built for the iPRoBeLab research pipeline, it lets you step through thousands of image pairs and inspect per-claim visual support scores, accuracy scores, and consistency check results side by side with the actual face images.

---

## Architecture

```
claim-viewer/
├── server.js          # Express backend (port 3001)
└── src/
    └── App.js         # React frontend (port 3000)
```

The React app proxies all `/api/*` requests to the Express server via CRA's built-in proxy. Two processes must run simultaneously.

---

## Running

**Terminal 1 — backend:**
```bash
cd claim-viewer
node server.js
```

**Terminal 2 — React frontend:**
```bash
cd claim-viewer
npm start
```

Then open `http://localhost:3000` in your browser.

---

## Data Paths

Configured in `server.js`:

| Constant | Path |
|---|---|
| `DATA_DIR` | `results/IJBS/Explanations-with-gt/gpt-4o/claims/Qwen2.5-7B-Instruct` |
| `CONSISTENCY_DIR` | `results/IJBS/Explanations-with-gt/gpt-4o/consistency/Qwen2.5-7B-Instruct` |
| `IMAGE_DIR` | `/mnt/research/iPRoBeLab/sonymd/LikelihoodRatio/.data/IJBS/IJBS-Still` |

To point the viewer at a different model or dataset, update these three constants and restart the server.

---

## JSONL Format

### Claims file (`pair_XXXXX.jsonl`)

**Line 1 — metadata:**
```json
{"file_id": "pair_00000", "status": "success", "n_claims": 10, "image1": "1/img_101146.jpg", "image2": "1/img_101147.jpg", "label": "1"}
```

**Lines 2+ — claims:**
```json
{"file_id": "pair_00000", "claim_id": "C01", "claim_text": "...", "claim_type": "difference", "feature_category": "eyes", "image_reference": "both", "specificity": "precise", "visual_support": 4, "visual_accuracy": 3, "support_rationale": "...", "accuracy_rationale": "...", "judge_model": "..."}
```

### Consistency file (`pair_XXXXX.jsonl`)

**Line 1 — metadata:**
```json
{"pair_id": "pair_00000", "status": "success", "n_contradiction_pairs": 0, "n_contradictions_found": 0, "verdict_alignment": "ENTAILMENT", "image1": "...", "image2": "..."}
```

**Lines 2+ — checks:**
```json
{"check_type": "verdict_alignment", "verdict": "Non-match", "reasoning": "...", "label": "ENTAILMENT", "confidence": 5, "rationale": "..."}
```

---

## Features

### Image Pair Display
Both face images for the selected pair are shown side by side at the top of the page, served directly from `IMAGE_DIR` on disk. Each panel shows the relative file path as a caption.

### Claim Cards
Each claim is displayed as an expandable card showing:
- **Claim ID**, **type** (`similarity` / `difference` / `reasoning`), **specificity** (`vague` / `moderate` / `precise`), and **image reference** (`image_1` / `image_2` / `both`)
- **Visual Support** and **Visual Accuracy** score bars (out of 5), color-coded by metric
- **Feature category** indicated by icon and left-border color

Clicking a card expands it to reveal:
- Support rationale and accuracy rationale in a two-column layout
- Consistency overlay (if consistency data exists) with alignment label, confidence, and rationale
- Judge model identifier

### Feature Category Breakdown
A heatmap of all feature categories present in the current pair (e.g. eyes, nose, facial structure, skin texture). Each tile shows the claim count and average visual support score. Clicking a tile filters the claim list to that category; clicking again clears the filter.

### Claim Filtering and Sorting
- **Filter by type:** ALL / SIMILARITY / DIFFERENCE / REASONING
- **Filter by category:** click any tile in the category heatmap
- **Sort by:** Claim ID / Visual Support / Visual Accuracy
- Live count of currently shown claims

### Label Filter
Filter the entire file list by ground-truth pair label before selecting a file:
- **ALL** — show all pairs
- **SAME** — show only genuine pairs (label = 1)
- **DIFF** — show only impostor pairs (label = 0)

Labels are sourced from the background metadata index (see below) and work in combination with the search box.

### Search / Jump
A search box filters the file dropdown in real time by pair ID substring. Press `Enter` to immediately jump to the first match in the filtered list. Search and label filter are applied together.

### Keyboard Navigation
Use the `←` and `→` arrow keys to step through pairs one by one without touching the mouse. Navigation respects the active label filter and search query. Arrow keys are disabled while the search box has focus so typing is unaffected.

### Pair Navigation Bar
- Dropdown selector listing all currently visible files (post-filter + post-search)
- `← prev` / `next →` buttons
- `X / N` position counter within the filtered list
- Keyboard shortcut hint displayed inline

### Stats Row
Four summary cards for the currently loaded pair:
- **Total Claims** — with ground-truth label (same / different)
- **Avg Visual Support** — mean support score across all claims
- **Avg Visual Accuracy** — mean accuracy score across all claims
- **Precise Claims** — count and percentage with `precise` specificity

### Consistency Check Panel
When a consistency file exists for the selected pair, a summary panel appears above the claim list showing:
- **Verdict Alignment** — `ENTAILMENT` / `CONTRADICTION` / `NEUTRAL`, color-coded green / red / amber
- **Contradiction count** — number of contradictions found and contradiction pairs checked
- **Rationale** — the judge's reasoning for the verdict alignment with confidence score

The verdict alignment label is also echoed in the page header. Each claim card shows an inline consistency badge, and the full per-claim rationale is visible in the expanded view. The panel is hidden gracefully when no consistency data exists for a pair.

### Background Metadata Index
On startup the server streams only the first line of every `.jsonl` file in `DATA_DIR` to build a lightweight in-memory index of `{ label, image1, image2 }` per file. This powers the label filter with no pre-processing step. The frontend polls `/api/meta` every 3 seconds until the server signals the index is complete.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/files` | Sorted list of all `.jsonl` filenames in `DATA_DIR` |
| `GET` | `/api/meta` | Metadata index `{ ready, index }` built at startup |
| `GET` | `/api/files/:filename` | Raw content of a claims JSONL file |
| `GET` | `/api/consistency/:filename` | Raw content of a consistency JSONL file |
| `GET` | `/api/images/*` | Serves an image by relative path within `IMAGE_DIR` |
