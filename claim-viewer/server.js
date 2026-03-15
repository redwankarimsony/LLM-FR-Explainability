const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 3001;

const DATA_DIR = path.resolve(
  __dirname,
  "../results/IJBS/Explanations-with-gt/gpt-4o/claims/Qwen2.5-7B-Instruct"
);

const CONSISTENCY_DIR = path.resolve(
  __dirname,
  "../results/IJBS/Explanations-with-gt/gpt-4o/consistency/Qwen2.5-7B-Instruct"
);

const IMAGE_DIR = "/mnt/research/iPRoBeLab/sonymd/LikelihoodRatio/.data/IJBS/IJBS-Still";

// Build a metadata index (label, image1, image2) from first line of every file.
// Runs in background after startup.
let metaIndex = {}; // filename -> { label, image1, image2 }
let metaReady = false;

function buildMetaIndex() {
  fs.readdir(DATA_DIR, (err, files) => {
    if (err) { console.error("Meta index error:", err.message); return; }
    const jsonlFiles = files.filter(f => f.endsWith(".jsonl")).sort();
    let remaining = jsonlFiles.length;
    if (remaining === 0) { metaReady = true; return; }

    jsonlFiles.forEach(filename => {
      const filePath = path.join(DATA_DIR, filename);
      // Read only the first line efficiently
      const stream = fs.createReadStream(filePath, { encoding: "utf8" });
      let buf = "";
      stream.on("data", chunk => {
        buf += chunk;
        const nl = buf.indexOf("\n");
        if (nl !== -1) {
          stream.destroy();
          try {
            const meta = JSON.parse(buf.slice(0, nl));
            metaIndex[filename] = {
              label: meta.label,
              image1: meta.image1,
              image2: meta.image2,
            };
          } catch (_) {}
        }
      });
      stream.on("close", () => {
        remaining--;
        if (remaining === 0) { metaReady = true; console.log("Meta index ready."); }
      });
      stream.on("error", () => {
        remaining--;
        if (remaining === 0) { metaReady = true; }
      });
    });
  });
}

// --- Endpoints ---

app.get("/api/files", (req, res) => {
  fs.readdir(DATA_DIR, (err, files) => {
    if (err) return res.status(500).json({ error: err.message });
    const jsonlFiles = files.filter(f => f.endsWith(".jsonl")).sort();
    res.json(jsonlFiles);
  });
});

app.get("/api/meta", (req, res) => {
  res.json({ ready: metaReady, index: metaIndex });
});

app.get("/api/files/:filename", (req, res) => {
  const filename = path.basename(req.params.filename);
  const filePath = path.join(DATA_DIR, filename);
  if (!filePath.startsWith(DATA_DIR)) return res.status(400).json({ error: "Invalid filename" });
  fs.readFile(filePath, "utf8", (err, data) => {
    if (err) return res.status(404).json({ error: "File not found" });
    res.type("text/plain").send(data);
  });
});

app.get("/api/consistency/:filename", (req, res) => {
  const filename = path.basename(req.params.filename);
  const filePath = path.join(CONSISTENCY_DIR, filename);
  if (!filePath.startsWith(CONSISTENCY_DIR)) return res.status(400).json({ error: "Invalid filename" });
  fs.readFile(filePath, "utf8", (err, data) => {
    if (err) return res.status(404).json({ error: "Not found" });
    res.type("text/plain").send(data);
  });
});

app.get("/api/images/*", (req, res) => {
  const relativePath = req.params[0];
  const filePath = path.join(IMAGE_DIR, relativePath);
  if (!filePath.startsWith(IMAGE_DIR)) return res.status(400).json({ error: "Invalid path" });
  res.sendFile(filePath, err => {
    if (err) res.status(404).json({ error: "Image not found" });
  });
});

app.listen(PORT, () => {
  console.log(`Data server running at http://localhost:${PORT}`);
  console.log(`Serving files from: ${DATA_DIR}`);
  console.log("Building metadata index in background...");
  buildMetaIndex();
});
