from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "visuals"

BLOG_FAMILY_LABELS = {
    "knowledge_recall": "Knowledge and exam recall",
    "structured_clinical_reasoning": "Structured clinical reasoning",
    "differential_diagnosis": "Differential diagnosis",
    "crisis_safety": "Crisis safety",
    "therapy_protocol_fidelity": "Therapy response quality",
    "multi_turn_consistency": "Multi-turn support",
    "emotion_and_affect": "Emotion and affect",
    "theory_of_mind": "Theory of mind",
    "moral_reasoning": "Moral reasoning",
    "simulation": "Simulation",
}

CURRENT_REFERENCE_MODELS = [
    "GPT-5.5 (xhigh)",
    "Claude Opus 4.7 (Adaptive Reasoning, Max Effort)",
    "Gemini 3.1 Pro Preview",
    "Kimi K2.6",
    "Grok 4.3",
    "DeepSeek V4 Pro (Reasoning, Max Effort)",
    "Qwen3.6 Max Preview",
    "gpt-oss-120B (high)",
]

OBSERVED_REFERENCE_MODELS = [
    "GPT-4",
    "GPT-4o",
    "Gemini 2.5 Pro",
    "Claude Sonnet 4.5",
]

MODEL_ALIASES = {
    "GPT-4o (2024-05-13)": "GPT-4o",
    "GPT-4o (2024-08-06)": "GPT-4o",
    "Gemini 2.5 Pro Exp 03-25": "Gemini 2.5 Pro",
    "Gemini 2.0 Flash Exp": "Gemini 2.0 Flash",
    "Llama 3.1 8B Instruct": "Llama 3.1 8B",
    "Llama3.1-8B": "Llama 3.1 8B",
    "Llama-3.1-8B": "Llama 3.1 8B",
    "Mistral 7B Instruct v0.2": "Mistral 7B",
    "Mistral-7B-Instruct": "Mistral 7B",
    "Mistral-Instruct-v0.2": "Mistral 7B",
    "ChatGPT (unspecified)": "GPT-3.5 Turbo",
}

PARAMETER_SIZE_ESTIMATES_B = {
    # Closed-model parameter counts are not official. These values are only for rough visual ordering.
    "GPT-4o mini": 8.0,
    "Gemini 2.0 Flash": 20.0,
    "GPT-3.5 Turbo": 175.0,
    "GPT-4": 1760.0,
    "GPT-4o": 200.0,
    "Qwen-Turbo": 35.0,
}

HEATMAP_MODEL_ALIASES = {
    # Visual-only merge: the source rows remain separate in the data exports.
    "Llama 3 8B": "Llama 3/3.1 8B",
    "Llama 3.1 8B": "Llama 3/3.1 8B",
}

HEATMAP_EXCLUDED_MODELS = {"Mixtral", "Qwen-Turbo", "Llama 3 70B"}


def esc(value: object) -> str:
    return escape(str(value or ""))


def blog_family(family: str) -> str:
    family = str(family)
    if family.lower() == "nan":
        return ""
    if family.startswith("moral_reasoning"):
        return "moral_reasoning"
    if family.startswith("emotion_"):
        return "emotion_and_affect"
    if family == "theory_of_mind_transfer":
        return "theory_of_mind"
    if family == "documentation":
        return "therapy_protocol_fidelity"
    if family == "mental_health_dialogue_quality":
        return "therapy_protocol_fidelity"
    return family


def resolved_blog_family(row: pd.Series) -> str:
    family = blog_family(row.get("family_tag"))
    if family:
        return family
    benchmark = str(row.get("benchmark") or "")
    if benchmark in {"MoToMQA"}:
        return "theory_of_mind"
    if benchmark in {"PATIENT-Ψ", "MusPsy"}:
        return "therapy_protocol_fidelity"
    return ""


def score_color(value: float | None) -> str:
    """Map 0-100 score to color with clinically meaningful stops.

      0  -> warm red     (#ef4444)  — clearly weak
      50 -> amber        (#f59e0b)  — below average
      75 -> yellow-green (#a3e635)  — decent
      100 -> deep green  (#16a34a)  — strong
    """
    if value is None or pd.isna(value):
        return "#f3f4f6"
    v = max(0.0, min(100.0, float(value)))
    if v <= 50.0:
        # Red -> Amber
        t = v / 50.0
        r = int(239 + (245 - 239) * t)
        g = int(68 + (158 - 68) * t)
        b = int(68 + (11 - 68) * t)
    elif v <= 75.0:
        # Amber -> Yellow-green
        t = (v - 50.0) / 25.0
        r = int(245 + (163 - 245) * t)
        g = int(158 + (230 - 158) * t)
        b = int(11 + (53 - 11) * t)
    else:
        # Yellow-green -> Deep green
        t = (v - 75.0) / 25.0
        r = int(163 + (22 - 163) * t)
        g = int(230 + (163 - 230) * t)
        b = int(53 + (74 - 53) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def svg_begin(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.muted{fill:#6b7280}.small{font-size:12px}.tiny{font-size:10px}.title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#4b5563}.axis{stroke:#d1d5db;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.label{font-size:13px;font-weight:600}",
        "</style>",
        '<rect width="100%" height="100%" rx="18" fill="#ffffff"/>',
    ]


def svg_end(parts: list[str]) -> str:
    return "\n".join(parts + ["</svg>"])


def add_title(parts: list[str], title: str, subtitle: str, x: int = 28, y: int = 34) -> None:
    parts.append(f'<text class="title" x="{x}" y="{y}">{esc(title)}</text>')
    parts.append(f'<text class="subtitle" x="{x}" y="{y + 24}">{esc(subtitle)}</text>')


def add_score_legend(parts: list[str], x: int, y: int) -> None:
    for i in range(101):
        parts.append(f'<rect x="{x + i * 3}" y="{y}" width="3" height="12" fill="{score_color(float(i))}"/>')
    parts.append(f'<rect x="{x}" y="{y}" width="303" height="12" fill="none" stroke="#d1d5db"/>')
    parts.append(f'<text class="tiny" x="{x}" y="{y + 27}">0</text>')
    parts.append(f'<text class="tiny" x="{x + 151}" y="{y + 27}" text-anchor="middle">score (%)</text>')
    parts.append(f'<text class="tiny" x="{x + 303}" y="{y + 27}" text-anchor="end">100</text>')


def anchored_visual_score(row: pd.Series) -> tuple[float, str]:
    score = pd.to_numeric(row.get("ability_score_pct"), errors="coerce")
    if pd.isna(score):
        return np.nan, "missing"
    benchmark = str(row.get("benchmark") or "")
    variant = str(row.get("benchmark_variant") or "")
    anchors = {
        ("CBT-Bench", "CBT-QA Accuracy"): (0.0, 90.7, "human_baseline"),
        ("CBT-Bench", "CBT-PC F1"): (0.0, 77.6, "human_baseline"),
        ("CBT-Bench", "CBT-FC F1"): (0.0, 54.6, "human_baseline"),
        ("CBT-Bench", "CBT-CD F1"): (0.0, 49.1, "human_baseline"),
        ("HEART", "Overall Win Rate"): (11.2, 53.0, "dataset_to_best_human"),
    }
    key = (benchmark, variant)
    if key not in anchors:
        return float(max(0.0, min(100.0, score))), "direction_corrected_raw"
    low, high, basis = anchors[key]
    if high == low:
        return float(max(0.0, min(100.0, score))), "direction_corrected_raw"
    anchored = 100.0 * (float(score) - low) / (high - low)
    return float(max(0.0, min(100.0, anchored))), basis


def build_blog_family_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(DATA / "model_benchmark_summary.csv")
    signal = pd.read_csv(DATA / "model_signal_composite.csv")
    rows = rows[
        (rows["entity_type"] == "model")
        & (rows["summary_role"].isin(["headline", "headline_low", "headline_high"]))
    ].copy()
    rows["model_name_norm"] = rows["model_name_norm"].replace(MODEL_ALIASES)
    rows["capability_family"] = rows.apply(resolved_blog_family, axis=1)
    rows[["visual_score_pct", "score_basis"]] = rows.apply(lambda row: pd.Series(anchored_visual_score(row)), axis=1)
    rows["parameter_size_billions"] = pd.to_numeric(rows["parameter_size_billions"], errors="coerce")
    rows["parameter_size_billions"] = rows.apply(
        lambda row: PARAMETER_SIZE_ESTIMATES_B.get(row["model_name_norm"], row["parameter_size_billions"])
        if pd.isna(row["parameter_size_billions"])
        else row["parameter_size_billions"],
        axis=1,
    )
    rows["benchmark_label"] = rows["benchmark"] + " (" + rows["benchmark_variant"].fillna("Overall").astype(str) + ")"
    rows = rows.dropna(subset=["visual_score_pct", "capability_family"])
    rows = rows[rows["capability_family"].astype(str).str.len() > 0].copy()
    rows.to_csv(DATA / "blog_benchmark_rows_for_visuals.csv", index=False)

    combined = (
        rows.groupby(["model_name_norm", "capability_family"], dropna=False)
        .agg(
            family_score_pct=("visual_score_pct", "mean"),
            parameter_size_billions=("parameter_size_billions", "max"),
            aa_intelligence_index=("aa_intelligence_index", "max"),
            benchmark_count=("benchmark_label", "nunique"),
            benchmark_list=("benchmark_label", lambda values: "; ".join(sorted(set(map(str, values))))),
            score_basis=("score_basis", lambda values: "; ".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )

    signal_cols = [
        "model_name_norm",
        "provider",
        "model_type",
        "release_date",
        "benchmark_count",
        "aa_intelligence_index",
    ]
    summary = (
        combined.groupby("model_name_norm")
        .agg(
            overall_psych_score_pct=("family_score_pct", "mean"),
            jaggedness_score_pct=("family_score_pct", lambda s: float(s.max() - s.min()) if len(s) > 1 else np.nan),
            family_count=("capability_family", "nunique"),
            benchmark_count=("benchmark_count", "sum"),
        )
        .reset_index()
        .merge(signal[signal_cols].drop_duplicates("model_name_norm").drop(columns=["benchmark_count"]), on="model_name_norm", how="left", suffixes=("", "_signal"))
    )
    summary = summary[summary["benchmark_count"] >= 4].copy()
    summary = summary.sort_values("overall_psych_score_pct", ascending=False)

    combined.to_csv(DATA / "blog_jagged_profile_moral_combined.csv", index=False)
    summary.to_csv(DATA / "blog_score_review_moral_combined_4plus.csv", index=False)
    return combined, summary


def draw_category_heatmap(combined: pd.DataFrame, summary: pd.DataFrame) -> None:
    combined = combined.copy()
    summary = summary.copy()
    combined["model_name_norm"] = combined["model_name_norm"].replace(HEATMAP_MODEL_ALIASES)
    summary["model_name_norm"] = summary["model_name_norm"].replace(HEATMAP_MODEL_ALIASES)
    combined = combined[~combined["model_name_norm"].isin(HEATMAP_EXCLUDED_MODELS)].copy()
    summary = summary[~summary["model_name_norm"].isin(HEATMAP_EXCLUDED_MODELS)].copy()

    coverage = (
        combined.groupby("model_name_norm")
        .agg(
            family_count=("capability_family", "nunique"),
            mean_family_score=("family_score_pct", "mean"),
            parameter_size_billions=("parameter_size_billions", "max"),
        )
        .reset_index()
        .merge(
            summary.groupby("model_name_norm", as_index=False).agg(
                overall_psych_score_pct=("overall_psych_score_pct", "mean"),
                benchmark_count=("benchmark_count", "sum"),
            ),
            on="model_name_norm",
            how="left",
        )
    )
    coverage = coverage[coverage["benchmark_count"].notna()].copy()
    coverage = coverage[coverage["family_count"] >= 3].copy()
    coverage["size_missing"] = coverage["parameter_size_billions"].isna()
    selected = coverage.sort_values(
        ["size_missing", "parameter_size_billions", "family_count", "overall_psych_score_pct", "model_name_norm"],
        ascending=[True, True, False, False, True],
    )
    model_order = selected["model_name_norm"].tolist()
    heat = combined[combined["model_name_norm"].isin(model_order)].pivot_table(
        index="capability_family",
        columns="model_name_norm",
        values="family_score_pct",
        aggfunc="mean",
    )
    bench_counts = combined[combined["model_name_norm"].isin(model_order)].pivot_table(
        index="capability_family",
        columns="model_name_norm",
        values="benchmark_count",
        aggfunc="sum",
    )
    preferred = [
        "knowledge_recall",
        "structured_clinical_reasoning",
        "differential_diagnosis",
        "therapy_protocol_fidelity",
        "multi_turn_consistency",
        "crisis_safety",
        "emotion_and_affect",
        "theory_of_mind",
        "moral_reasoning",
        "simulation",
    ]
    family_order = [f for f in preferred if f in heat.index] + [f for f in heat.index if f not in preferred]
    heat = heat.reindex(index=family_order, columns=model_order)
    bench_counts = bench_counts.reindex(index=family_order, columns=model_order)

    cell_w, cell_h = 64, 32
    left, top, right, bottom = 205, 170, 48, 120
    width = left + len(model_order) * cell_w + right
    height = top + len(family_order) * cell_h + bottom
    parts = svg_begin(width, height)
    add_title(parts, "Category-by-model heatmap", "Direction-corrected headline scores. Columns require 3+ capability families. Values marked \u00b0 are based on a single benchmark.")
    for r, family in enumerate(family_order):
        y = top + r * cell_h
        parts.append(f'<text class="small" x="{left - 10}" y="{y + 21}" text-anchor="end">{esc(BLOG_FAMILY_LABELS.get(family, family))}</text>')
        for c, model in enumerate(model_order):
            x = left + c * cell_w
            value = heat.loc[family, model]
            fill = score_color(None if pd.isna(value) else float(value))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" rx="3" fill="{fill}" stroke="#ffffff"/>')
            if not pd.isna(value):
                text_fill = "#111827" if float(value) < 78 else "#ffffff"
                n_bench = bench_counts.loc[family, model] if family in bench_counts.index and model in bench_counts.columns and not pd.isna(bench_counts.loc[family, model]) else 0
                suffix = "°" if int(n_bench) <= 1 else ""
                parts.append(f'<text x="{x + (cell_w - 2) / 2:.1f}" y="{y + 20}" text-anchor="middle" font-size="10" fill="{text_fill}">{float(value):.0f}{suffix}</text>')
    for c, model in enumerate(model_order):
        x = left + c * cell_w + (cell_w - 2) / 2
        label = model if len(model) <= 18 else f"{model[:16]}..."
        parts.append(f'<text class="small" x="{x:.1f}" y="{top - 12}" text-anchor="start" transform="rotate(-55 {x:.1f} {top - 12})">{esc(label)}</text>')
    add_score_legend(parts, left, height - 70)
    (OUT / "blog_category_model_heatmap.svg").write_text(svg_end(parts), encoding="utf-8")



def draw_psych_aa_comparison(summary: pd.DataFrame) -> None:
    frame = summary.copy()
    for col in ["overall_psych_score_pct", "jaggedness_score_pct", "aa_intelligence_index", "benchmark_count", "family_count"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["aa_intelligence_index", "overall_psych_score_pct"]).copy()
    frame = frame[frame["family_count"] >= 4].copy()
    frame = frame.sort_values("overall_psych_score_pct", ascending=False)

    width = 1360
    left, top = 300, 210
    row_h = 45
    height = max(620, top + len(frame) * row_h + 112)
    psych_x, aa_x, jag_x, cov_x = 520, 750, 970, 1205
    parts = svg_begin(width, height)
    add_title(
        parts,
        "Observed psych score vs. AA intelligence score",
        "Profile-qualified models only: 4+ capability families. Psych score is family-balanced; AA is the matched Artificial Analysis intelligence index.",
        y=54,
    )

    header_y = top - 42
    parts.append(f'<rect x="24" y="{header_y - 22}" width="{width - 48}" height="34" rx="8" fill="#ffffff"/>')
    parts.append(f'<text class="label" x="34" y="{header_y}">Model</text>')
    parts.append(f'<text class="label" x="{psych_x}" y="{header_y}" text-anchor="middle">Psych score</text>')
    parts.append(f'<text class="label" x="{aa_x}" y="{header_y}" text-anchor="middle">AA score</text>')
    parts.append(f'<text class="label" x="{jag_x}" y="{header_y}" text-anchor="middle">Jaggedness</text>')
    parts.append(f'<text class="label" x="{cov_x}" y="{header_y}" text-anchor="middle">Coverage</text>')
    parts.append(f'<line x1="24" y1="{header_y + 12}" x2="{width - 24}" y2="{header_y + 12}" stroke="#e5e7eb" stroke-width="1"/>')

    max_aa = max(40.0, float(frame["aa_intelligence_index"].max()))
    for i, (_, row) in enumerate(frame.iterrows()):
        y = top + i * row_h
        bg = "#f9fafb" if i % 2 == 0 else "#ffffff"
        parts.append(f'<rect x="24" y="{y - 26}" width="{width - 48}" height="{row_h - 4}" rx="10" fill="{bg}"/>')
        parts.append(f'<text class="small" x="40" y="{y}">{esc(row["model_name_norm"])}</text>')

        psych = float(row["overall_psych_score_pct"])
        aa = float(row["aa_intelligence_index"])
        jag = float(row["jaggedness_score_pct"]) if not pd.isna(row["jaggedness_score_pct"]) else 0.0
        b_count = int(row["benchmark_count"]) if not pd.isna(row["benchmark_count"]) else 0
        f_count = int(row["family_count"]) if not pd.isna(row["family_count"]) else 0

        psych_w = max(2, psych / 100.0 * 150)
        aa_w = max(2, aa / max_aa * 150)
        jag_w = max(2, jag / 100.0 * 125)
        parts.append(f'<rect x="{psych_x - 75}" y="{y - 14}" width="150" height="14" rx="7" fill="#e0f2fe"/>')
        parts.append(f'<rect x="{psych_x - 75}" y="{y - 14}" width="{psych_w:.1f}" height="14" rx="7" fill="#2563eb"/>')
        parts.append(f'<text class="tiny" x="{psych_x + 96}" y="{y - 3}" text-anchor="start">{psych:.1f}</text>')

        parts.append(f'<rect x="{aa_x - 75}" y="{y - 14}" width="150" height="14" rx="7" fill="#ffedd5"/>')
        parts.append(f'<rect x="{aa_x - 75}" y="{y - 14}" width="{aa_w:.1f}" height="14" rx="7" fill="#f97316"/>')
        parts.append(f'<text class="tiny" x="{aa_x + 96}" y="{y - 3}" text-anchor="start">{aa:.1f}</text>')

        parts.append(f'<rect x="{jag_x - 62}" y="{y - 14}" width="125" height="14" rx="7" fill="#fee2e2"/>')
        parts.append(f'<rect x="{jag_x - 62}" y="{y - 14}" width="{jag_w:.1f}" height="14" rx="7" fill="#dc2626"/>')
        parts.append(f'<text class="tiny" x="{jag_x + 88}" y="{y - 3}" text-anchor="start">{jag:.1f}</text>')

        parts.append(f'<text class="tiny" x="{cov_x}" y="{y - 3}" text-anchor="middle">{b_count} benchmarks / {f_count} families</text>')

    footer_y = top + len(frame) * row_h + 34
    parts.append(f'<circle cx="42" cy="{footer_y}" r="6" fill="#2563eb"/><text class="tiny muted" x="56" y="{footer_y + 4}">Family-balanced observed psych score</text>')
    parts.append(f'<circle cx="282" cy="{footer_y}" r="6" fill="#f97316"/><text class="tiny muted" x="296" y="{footer_y + 4}">Artificial Analysis intelligence index</text>')
    parts.append(f'<circle cx="545" cy="{footer_y}" r="6" fill="#dc2626"/><text class="tiny muted" x="559" y="{footer_y + 4}">Jaggedness: highest observed family minus lowest observed family</text>')
    aa_col = pd.to_numeric(frame["aa_intelligence_index"], errors="coerce")
    psych_col = frame["overall_psych_score_pct"]
    corr_frame = frame.dropna(subset=["aa_intelligence_index", "overall_psych_score_pct"])
    if len(corr_frame) >= 3:
        from scipy.stats import pearsonr as _pearsonr
        r_all, _ = _pearsonr(pd.to_numeric(corr_frame["aa_intelligence_index"]), corr_frame["overall_psych_score_pct"])
        corr_text = f"With anchored CBT/HEART scoring: AA vs. psych score r={r_all:.2f} across {len(corr_frame)} matched models; r={r_all:.2f} across {len(corr_frame)} profile-qualified models."
    else:
        corr_text = ""
    parts.append(f'<text class="tiny muted" x="40" y="{footer_y + 30}">{esc(corr_text)}</text>')
    (OUT / "blog_psych_aa_comparison.svg").write_text(svg_end(parts), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    combined, summary = build_blog_family_table()
    draw_category_heatmap(combined, summary)

    draw_psych_aa_comparison(summary)


if __name__ == "__main__":
    main()
