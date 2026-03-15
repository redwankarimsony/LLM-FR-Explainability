import { useState, useMemo, useEffect, useCallback, useRef } from "react";

function parseJSONL(raw) {
  const lines = raw.trim().split("\n").filter(Boolean);
  const meta = JSON.parse(lines[0]);
  const claims = lines.slice(1).map(l => JSON.parse(l));
  return { meta, claims };
}

const CATEGORY_ICONS = {
  facial_structure: "◈", eyes: "◉", nose: "△", mouth_lips: "◡",
  eyebrows: "⌒", skin_texture: "▦", periocular: "◎", ears: "◁",
  hair_hairline: "⌇", lighting_background: "☀", expression: "◟",
  age_and_appearance: "◷", overall_reasoning: "⬡",
};

const CATEGORY_COLORS = {
  facial_structure: "#6EE7B7", eyes: "#93C5FD", nose: "#FCA5A5",
  mouth_lips: "#F9A8D4", eyebrows: "#C4B5FD", skin_texture: "#FDE68A",
  periocular: "#A5F3FC", ears: "#86EFAC", hair_hairline: "#FED7AA",
  lighting_background: "#E9D5FF", expression: "#99F6E4",
  age_and_appearance: "#FEF08A", overall_reasoning: "#CBD5E1",
};

function ScoreBar({ value, max = 5, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 4, background: "#1e293b", borderRadius: 2 }}>
        <div style={{ width: `${(value / max) * 100}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ fontFamily: "monospace", fontSize: 11, color, minWidth: 20 }}>{value}/{max}</span>
    </div>
  );
}

function SpecBadge({ value }) {
  const colors = { vague: "#94a3b8", moderate: "#f59e0b", precise: "#10b981" };
  return (
    <span style={{ fontSize: 9, fontFamily: "monospace", letterSpacing: 1, color: colors[value] || "#94a3b8", border: `1px solid ${colors[value] || "#94a3b8"}`, borderRadius: 2, padding: "1px 5px", textTransform: "uppercase" }}>
      {value}
    </span>
  );
}

function ClaimCard({ claim, consistencyMap }) {
  const [open, setOpen] = useState(false);
  const catColor = CATEGORY_COLORS[claim.feature_category] || "#CBD5E1";
  const icon = CATEGORY_ICONS[claim.feature_category] || "●";
  const typeColor = { similarity: "#10b981", difference: "#f43f5e", reasoning: "#8b5cf6" }[claim.claim_type] || "#94a3b8";
  const consistency = consistencyMap?.[claim.claim_id];

  const alignColor = { ENTAILMENT: "#10b981", CONTRADICTION: "#f43f5e", NEUTRAL: "#f59e0b" };

  return (
    <div onClick={() => setOpen(!open)} style={{
      background: "#0f172a", border: `1px solid ${open ? catColor + "55" : "#1e293b"}`,
      borderLeft: `3px solid ${catColor}`, borderRadius: 8, padding: "12px 16px",
      cursor: "pointer", transition: "all 0.2s", marginBottom: 8,
    }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        <span style={{ color: catColor, fontSize: 16, marginTop: 1, flexShrink: 0 }}>{icon}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, flexWrap: "wrap" }}>
            <span style={{ fontFamily: "monospace", fontSize: 10, color: "#475569" }}>{claim.claim_id}</span>
            <span style={{ fontSize: 9, color: typeColor, border: `1px solid ${typeColor}55`, borderRadius: 2, padding: "1px 5px", textTransform: "uppercase", letterSpacing: 1 }}>{claim.claim_type}</span>
            <SpecBadge value={claim.specificity} />
            <span style={{ fontSize: 9, color: "#475569", fontFamily: "monospace" }}>{claim.image_reference}</span>
            {consistency && (
              <span style={{ fontSize: 9, color: alignColor[consistency.label] || "#94a3b8", border: `1px solid ${alignColor[consistency.label] || "#94a3b8"}55`, borderRadius: 2, padding: "1px 5px", textTransform: "uppercase", letterSpacing: 1 }}>
                {consistency.label}
              </span>
            )}
          </div>
          <p style={{ margin: 0, fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>{claim.claim_text}</p>
          <div style={{ display: "flex", gap: 20, marginTop: 8 }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 9, color: "#475569", marginBottom: 3, letterSpacing: 1 }}>SUPPORT</div>
              <ScoreBar value={claim.visual_support} color="#38bdf8" />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 9, color: "#475569", marginBottom: 3, letterSpacing: 1 }}>ACCURACY</div>
              <ScoreBar value={claim.visual_accuracy} color="#a78bfa" />
            </div>
          </div>
        </div>
      </div>

      {open && (
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid #1e293b" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div style={{ background: "#0a0f1a", borderRadius: 6, padding: 10 }}>
              <div style={{ fontSize: 9, color: "#38bdf8", letterSpacing: 1, marginBottom: 4 }}>SUPPORT RATIONALE</div>
              <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{claim.support_rationale}</p>
            </div>
            <div style={{ background: "#0a0f1a", borderRadius: 6, padding: 10 }}>
              <div style={{ fontSize: 9, color: "#a78bfa", letterSpacing: 1, marginBottom: 4 }}>ACCURACY RATIONALE</div>
              <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{claim.accuracy_rationale}</p>
            </div>
          </div>
          {consistency && (
            <div style={{ marginTop: 10, background: "#0a0f1a", borderRadius: 6, padding: 10, borderLeft: `3px solid ${alignColor[consistency.label] || "#475569"}` }}>
              <div style={{ fontSize: 9, color: alignColor[consistency.label] || "#94a3b8", letterSpacing: 1, marginBottom: 4 }}>
                CONSISTENCY · {consistency.label} · confidence {consistency.confidence}/5
              </div>
              <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{consistency.rationale}</p>
            </div>
          )}
          <div style={{ marginTop: 8, fontSize: 9, color: "#334155", fontFamily: "monospace" }}>judge: {claim.judge_model}</div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, padding: "14px 18px" }}>
      <div style={{ fontSize: 9, color: "#475569", letterSpacing: 2, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontFamily: "Georgia, serif", fontWeight: "bold", color: color || "#f1f5f9" }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: "#475569", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function App() {
  const [fileList, setFileList] = useState([]);
  const [metaIndex, setMetaIndex] = useState({});
  const [selectedFile, setSelectedFile] = useState("");
  const [rawData, setRawData] = useState(null);
  const [consistency, setConsistency] = useState(null); // parsed consistency JSONL
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [claimFilter, setClaimFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [sortBy, setSortBy] = useState("claim_id");
  const [labelFilter, setLabelFilter] = useState("all"); // "all" | "same" | "different"
  const [searchQuery, setSearchQuery] = useState("");

  const searchRef = useRef(null);

  // Load file list and meta index on mount
  useEffect(() => {
    fetch("/api/files")
      .then(r => r.json())
      .then(files => {
        setFileList(files);
        if (files.length > 0) setSelectedFile(files[0]);
      })
      .catch(err => setError("Could not load file list: " + err.message));

    // Poll meta index until ready
    const pollMeta = () => {
      fetch("/api/meta")
        .then(r => r.json())
        .then(({ ready, index }) => {
          setMetaIndex(index);
          if (!ready) setTimeout(pollMeta, 3000);
        })
        .catch(() => {});
    };
    pollMeta();
  }, []);

  // Load selected file content + consistency data
  useEffect(() => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setRawData(null);
    setConsistency(null);

    fetch(`/api/files/${encodeURIComponent(selectedFile)}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
      .then(text => { setRawData(text); setLoading(false); })
      .catch(err => { setError(err.message); setLoading(false); });

    fetch(`/api/consistency/${encodeURIComponent(selectedFile)}`)
      .then(r => r.ok ? r.text() : null)
      .then(text => {
        if (!text) return;
        const lines = text.trim().split("\n").filter(Boolean);
        const meta = JSON.parse(lines[0]);
        const checks = lines.slice(1).map(l => JSON.parse(l));
        setConsistency({ meta, checks });
      })
      .catch(() => {});
  }, [selectedFile]);

  const parsed = useMemo(() => {
    if (!rawData) return null;
    try { return parseJSONL(rawData); } catch { return null; }
  }, [rawData]);

  const { meta, claims } = parsed || { meta: null, claims: [] };

  // Build consistency map: claim_id -> check (for inline display)
  const consistencyMap = useMemo(() => {
    if (!consistency) return null;
    const map = {};
    consistency.checks.forEach(c => { if (c.claim_id) map[c.claim_id] = c; });
    return map;
  }, [consistency]);

  // Filtered file list (label + search)
  const visibleFiles = useMemo(() => {
    return fileList.filter(f => {
      const m = metaIndex[f];
      if (labelFilter !== "all" && m) {
        const isSame = String(m.label) === "1";
        if (labelFilter === "same" && !isSame) return false;
        if (labelFilter === "different" && isSame) return false;
      }
      if (searchQuery) {
        return f.toLowerCase().includes(searchQuery.toLowerCase());
      }
      return true;
    });
  }, [fileList, metaIndex, labelFilter, searchQuery]);

  const currentIdx = visibleFiles.indexOf(selectedFile);

  const goTo = useCallback((idx) => {
    if (idx < 0 || idx >= visibleFiles.length) return;
    setSelectedFile(visibleFiles[idx]);
    setClaimFilter("all");
    setCategoryFilter("all");
  }, [visibleFiles]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      // Don't fire when typing in the search box
      if (document.activeElement === searchRef.current) return;
      if (e.key === "ArrowLeft") goTo(currentIdx - 1);
      if (e.key === "ArrowRight") goTo(currentIdx + 1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [currentIdx, goTo]);

  const categories = [...new Set(claims.map(c => c.feature_category))];
  const avgSupport = claims.length ? (claims.reduce((s, c) => s + c.visual_support, 0) / claims.length).toFixed(2) : "—";
  const avgAccuracy = claims.length ? (claims.reduce((s, c) => s + c.visual_accuracy, 0) / claims.length).toFixed(2) : "—";
  const preciseCt = claims.filter(c => c.specificity === "precise").length;

  const filteredClaims = claims
    .filter(c => claimFilter === "all" || c.claim_type === claimFilter)
    .filter(c => categoryFilter === "all" || c.feature_category === categoryFilter)
    .sort((a, b) => {
      if (sortBy === "support") return b.visual_support - a.visual_support;
      if (sortBy === "accuracy") return b.visual_accuracy - a.visual_accuracy;
      return a.claim_id.localeCompare(b.claim_id);
    });

  const btnStyle = (active, accent) => ({
    background: active ? (accent ? accent + "22" : "#1e40af") : "#0f172a",
    border: `1px solid ${active ? (accent || "#3b82f6") : "#1e293b"}`,
    color: active ? (accent || "#93c5fd") : "#475569",
    borderRadius: 4, padding: "4px 10px",
    fontSize: 10, letterSpacing: 1, cursor: "pointer", textTransform: "uppercase",
  });

  const labelMeta = metaIndex[selectedFile];
  const pairLabel = labelMeta ? (String(labelMeta.label) === "1" ? "same" : "different") : null;
  const labelColor = pairLabel === "same" ? "#10b981" : pairLabel === "different" ? "#f43f5e" : "#475569";

  // Consistency summary
  const consistencyVerdict = consistency?.meta?.verdict_alignment;
  const verdictColor = { ENTAILMENT: "#10b981", CONTRADICTION: "#f43f5e", NEUTRAL: "#f59e0b" }[consistencyVerdict] || "#475569";

  return (
    <div style={{ minHeight: "100vh", background: "#020817", color: "#f1f5f9", fontFamily: "'Georgia', serif" }}>
      <style>{`* { box-sizing: border-box; } ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: #1e293b; } button { font-family: inherit; }`}</style>

      {/* Header */}
      <div style={{ borderBottom: "1px solid #0f172a", padding: "24px 32px", background: "#020817" }}>
        <div style={{ fontSize: 10, color: "#1d4ed8", letterSpacing: 3, marginBottom: 6 }}>FACE RECOGNITION EVALUATION</div>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <h1 style={{ margin: 0, fontSize: 28, fontWeight: "normal", color: "#f8fafc" }}>
            Claim Analysis {meta && <span style={{ color: "#334155", fontSize: 16 }}>/ {meta.file_id || meta.pair_id}</span>}
          </h1>
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            {meta && <>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981" }} />
                <span style={{ fontSize: 11, color: "#10b981", fontFamily: "monospace" }}>status: {meta.status}</span>
              </div>
            </>}
            {pairLabel && (
              <span style={{ fontSize: 11, fontFamily: "monospace", color: labelColor, border: `1px solid ${labelColor}55`, borderRadius: 4, padding: "2px 8px", textTransform: "uppercase", letterSpacing: 1 }}>
                {pairLabel}
              </span>
            )}
            {consistencyVerdict && (
              <span style={{ fontSize: 11, fontFamily: "monospace", color: verdictColor, border: `1px solid ${verdictColor}55`, borderRadius: 4, padding: "2px 8px", textTransform: "uppercase", letterSpacing: 1 }}>
                {consistencyVerdict}
              </span>
            )}
          </div>
        </div>

        {/* File selector row */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 16, flexWrap: "wrap" }}>
          {/* Label filter */}
          <span style={{ fontSize: 9, color: "#334155", letterSpacing: 2 }}>LABEL:</span>
          {[["all", "ALL", null], ["same", "SAME", "#10b981"], ["different", "DIFF", "#f43f5e"]].map(([v, l, c]) => (
            <button key={v} onClick={() => setLabelFilter(v)} style={btnStyle(labelFilter === v, c)}>{l}</button>
          ))}

          <div style={{ width: 1, height: 16, background: "#1e293b", margin: "0 4px" }} />

          {/* Search */}
          <input
            ref={searchRef}
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="search pair id…"
            style={{
              background: "#0f172a", border: "1px solid #1e293b", color: "#94a3b8",
              borderRadius: 4, padding: "4px 8px", fontSize: 11, fontFamily: "monospace",
              outline: "none", width: 160,
            }}
            onKeyDown={e => {
              if (e.key === "Enter" && visibleFiles.length > 0) {
                setSelectedFile(visibleFiles[0]);
                setClaimFilter("all"); setCategoryFilter("all");
              }
            }}
          />

          <div style={{ width: 1, height: 16, background: "#1e293b", margin: "0 4px" }} />

          {/* File dropdown */}
          <select
            value={selectedFile}
            onChange={e => { setSelectedFile(e.target.value); setClaimFilter("all"); setCategoryFilter("all"); }}
            style={{
              background: "#0f172a", border: "1px solid #1e293b", color: "#94a3b8",
              borderRadius: 4, padding: "4px 8px", fontSize: 11, fontFamily: "monospace",
              cursor: "pointer", minWidth: 200,
            }}
          >
            {visibleFiles.map(f => <option key={f} value={f}>{f}</option>)}
          </select>

          <span style={{ fontSize: 10, color: "#334155", fontFamily: "monospace" }}>
            {visibleFiles.length > 0 && `${currentIdx + 1} / ${visibleFiles.length}`}
          </span>

          <button disabled={currentIdx <= 0} onClick={() => goTo(currentIdx - 1)}
            style={{ ...btnStyle(false), opacity: currentIdx <= 0 ? 0.3 : 1 }}>← prev</button>
          <button disabled={currentIdx >= visibleFiles.length - 1} onClick={() => goTo(currentIdx + 1)}
            style={{ ...btnStyle(false), opacity: currentIdx >= visibleFiles.length - 1 ? 0.3 : 1 }}>next →</button>

          <span style={{ fontSize: 9, color: "#1e293b", fontFamily: "monospace", marginLeft: 4 }}>← → keys</span>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 24px" }}>

        {loading && (
          <div style={{ textAlign: "center", color: "#475569", padding: 60, fontFamily: "monospace", fontSize: 12 }}>
            loading {selectedFile}…
          </div>
        )}

        {error && (
          <div style={{ background: "#1a0a0a", border: "1px solid #7f1d1d", borderRadius: 8, padding: 16, color: "#fca5a5", fontFamily: "monospace", fontSize: 12 }}>
            error: {error}
          </div>
        )}

        {!loading && !error && meta && <>

          {/* Image pair */}
          {(meta.image1 || meta.image2) && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 28 }}>
              {[meta.image1, meta.image2].map((imgPath, i) => (
                <div key={i} style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, overflow: "hidden" }}>
                  <div style={{ fontSize: 9, color: "#475569", letterSpacing: 2, padding: "8px 12px", borderBottom: "1px solid #1e293b" }}>
                    IMAGE {i + 1} · <span style={{ color: "#334155", fontFamily: "monospace" }}>{imgPath}</span>
                  </div>
                  {imgPath ? (
                    <img src={`/api/images/${imgPath}`} alt={`image ${i + 1}`}
                      style={{ width: "100%", display: "block", objectFit: "contain", maxHeight: 320 }} />
                  ) : (
                    <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: "#334155", fontSize: 11 }}>no image</div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Stats row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 28 }}>
            <StatCard label="TOTAL CLAIMS" value={meta.n_claims}
              sub={pairLabel ? `label: ${pairLabel}` : undefined} color="#f1f5f9" />
            <StatCard label="AVG VISUAL SUPPORT" value={avgSupport} sub="out of 5.00" color="#38bdf8" />
            <StatCard label="AVG VISUAL ACCURACY" value={avgAccuracy} sub="out of 5.00" color="#a78bfa" />
            <StatCard label="PRECISE CLAIMS" value={preciseCt}
              sub={claims.length ? `${Math.round(preciseCt / claims.length * 100)}% of total` : undefined} color="#10b981" />
          </div>

          {/* Consistency summary panel */}
          {consistency && (
            <div style={{ background: "#0f172a", border: `1px solid ${verdictColor}33`, borderLeft: `3px solid ${verdictColor}`, borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontSize: 9, color: "#475569", letterSpacing: 2, marginBottom: 10 }}>CONSISTENCY CHECK</div>
              <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
                <div>
                  <div style={{ fontSize: 9, color: "#475569", letterSpacing: 1, marginBottom: 4 }}>VERDICT ALIGNMENT</div>
                  <span style={{ fontSize: 14, fontFamily: "monospace", color: verdictColor }}>{consistencyVerdict}</span>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "#475569", letterSpacing: 1, marginBottom: 4 }}>CONTRADICTIONS</div>
                  <span style={{ fontSize: 14, fontFamily: "monospace", color: consistency.meta.n_contradictions_found > 0 ? "#f43f5e" : "#10b981" }}>
                    {consistency.meta.n_contradictions_found} found · {consistency.meta.n_contradiction_pairs} pairs
                  </span>
                </div>
                {consistency.checks.filter(c => c.check_type === "verdict_alignment").map((c, i) => (
                  <div key={i} style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ fontSize: 9, color: "#475569", letterSpacing: 1, marginBottom: 4 }}>VERDICT · confidence {c.confidence}/5</div>
                    <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{c.rationale}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Category heatmap */}
          <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <div style={{ fontSize: 9, color: "#475569", letterSpacing: 2, marginBottom: 12 }}>FEATURE CATEGORY BREAKDOWN</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {categories.map(cat => {
                const catClaims = claims.filter(c => c.feature_category === cat);
                const avgS = (catClaims.reduce((s, c) => s + c.visual_support, 0) / catClaims.length).toFixed(1);
                const color = CATEGORY_COLORS[cat] || "#CBD5E1";
                return (
                  <div key={cat} onClick={e => { e.stopPropagation(); setCategoryFilter(categoryFilter === cat ? "all" : cat); }}
                    style={{ background: categoryFilter === cat ? color + "22" : "#0a0f1a", border: `1px solid ${categoryFilter === cat ? color : "#1e293b"}`, borderRadius: 6, padding: "8px 12px", cursor: "pointer", transition: "all 0.2s" }}>
                    <div style={{ fontSize: 9, color, marginBottom: 2 }}>{CATEGORY_ICONS[cat]} {cat.replace(/_/g, " ")}</div>
                    <div style={{ fontFamily: "monospace", fontSize: 11, color: "#94a3b8" }}>{catClaims.length} claims · {avgS} avg</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Filters */}
          <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 9, color: "#334155", letterSpacing: 2 }}>FILTER:</span>
            {["all", "similarity", "difference", "reasoning"].map(f => (
              <button key={f} onClick={() => setClaimFilter(f)} style={btnStyle(claimFilter === f)}>{f}</button>
            ))}
            <span style={{ fontSize: 9, color: "#334155", letterSpacing: 2, marginLeft: 12 }}>SORT:</span>
            {[["claim_id", "ID"], ["support", "Support"], ["accuracy", "Accuracy"]].map(([v, l]) => (
              <button key={v} onClick={() => setSortBy(v)} style={btnStyle(sortBy === v)}>{l}</button>
            ))}
            <span style={{ marginLeft: "auto", fontSize: 10, color: "#334155", fontFamily: "monospace" }}>{filteredClaims.length} claims shown</span>
          </div>

          {/* Claims */}
          <div>
            {filteredClaims.map(claim => <ClaimCard key={claim.claim_id} claim={claim} consistencyMap={consistencyMap} />)}
          </div>
        </>}
      </div>
    </div>
  );
}
