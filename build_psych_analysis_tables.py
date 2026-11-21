from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist, mean, median, pstdev
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def norm(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def key(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(text))


def parse_float(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        value = float(text)
        if math.isnan(value):
            return None
        return value
    cleaned = str(text).strip().replace("%", "").replace("−", "-").replace("–", "-")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
        if math.isnan(value):
            return None
        return value
    except ValueError:
        return None


def parse_int(text: Any) -> int | None:
    value = parse_float(text)
    if value is None or not value.is_integer():
        return None
    return int(value)


def release_cohort_fields(release_date: Any) -> dict[str, Any]:
    text = str(release_date or "").strip()
    if not text:
        return {
            "release_year": "",
            "release_month": "",
            "release_month_index": "",
            "release_month_key": "",
        }
    year = parse_int(text[:4])
    month = parse_int(text[5:7]) if len(text) >= 7 else None
    month_key = None
    month_index = None
    if year is not None:
        month_key = f"{year:04d}-{month:02d}" if month is not None else f"{year:04d}"
        if month is not None:
            month_index = year * 12 + month
    return {
        "release_year": year if year is not None else "",
        "release_month": month if month is not None else "",
        "release_month_index": month_index if month_index is not None else "",
        "release_month_key": month_key if month_key is not None else "",
    }


def load_csv(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / name
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_json(stem: str, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    csv_path = DATA_DIR / f"{stem}.csv"
    json_path = DATA_DIR / f"{stem}.json"
    
    def json_ready(value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        json_path.write_text("[]", encoding="utf-8")
        return
    if field_order is None:
        field_order = []
        seen: set[str] = set()
        for row in rows:
            for name in row.keys():
                if name not in seen:
                    seen.add(name)
                    field_order.append(name)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    cleaned_rows = [{k: json_ready(v) for k, v in row.items()} for row in rows]
    json_path.write_text(json.dumps(cleaned_rows, indent=2, ensure_ascii=False), encoding="utf-8")


BENCHMARK_META: dict[str, dict[str, Any]] = {
    "MentalBench": {
        "domain": "diagnosis",
        "task_type": "differential diagnosis",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "differential_diagnosis",
    },
    "MENTAT": {
        "domain": "clinical reasoning",
        "task_type": "real-world psychiatry",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "structured_clinical_reasoning",
    },
    "PsychiatryBench": {
        "domain": "clinical reasoning",
        "task_type": "multi-task psychiatry",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "structured_clinical_reasoning",
    },
    "PsychBench": {
        "domain": "clinical reasoning",
        "task_type": "adult psychiatry composite",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "structured_clinical_reasoning",
    },
    "PsychoBench": {
        "domain": "knowledge",
        "task_type": "counselor certification exam multiple choice",
        "metric_type": "accuracy and weighted classification metrics",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "medium",
        "family_tag": "knowledge_recall",
    },
    "CBT-Bench": {
        "domain": "therapy",
        "task_type": "CBT knowledge and classification",
        "metric_type": "accuracy and F1",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "therapy_protocol_fidelity",
    },
    "CBT-Bench Level III": {
        "domain": "therapy",
        "task_type": "counseling skill comparison",
        "metric_type": "pairwise score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "therapy_protocol_fidelity",
    },
    "CounselingBench": {
        "domain": "therapy",
        "task_type": "counseling competence",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "therapy_protocol_fidelity",
    },
    "MindEval": {
        "domain": "therapy",
        "task_type": "multi-turn mental health support",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "multi-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "therapy_protocol_fidelity",
    },
    "OMind-Chat": {
        "domain": "therapy",
        "task_type": "multi-turn mental health dialogue quality",
        "metric_type": "binary, scale, and Likert score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "multi-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "mental_health_dialogue_quality",
    },
    "HEART": {
        "domain": "therapy",
        "task_type": "multi-turn emotional support dialogue",
        "metric_type": "win rate and Elo",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "multi-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "high",
        "family_tag": "multi_turn_consistency",
    },
    "MoralBench": {
        "domain": "morality",
        "task_type": "moral judgment",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "low",
        "family_tag": "moral_reasoning_general",
    },
    "MoralChoice": {
        "domain": "morality",
        "task_type": "moral choice under ambiguity",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "high",
        "clinical_risk_level": "low",
        "family_tag": "moral_reasoning",
    },
    "EmoBench": {
        "domain": "emotion",
        "task_type": "emotion understanding/application",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "emotion",
    },
    "EmoBench-Reddit": {
        "domain": "emotion",
        "task_type": "hierarchical emotion understanding",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "emotion_understanding",
    },
    "EmoBench-M": {
        "domain": "emotion",
        "task_type": "foundational emotion recognition",
        "metric_type": "accuracy and weighted F1",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "emotion_recognition",
    },
    "MMLU-Pro": {
        "domain": "knowledge",
        "task_type": "graduate-level psychology multiple choice",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "low",
        "family_tag": "knowledge_recall",
    },
    "Theory of Mind": {
        "domain": "social cognition",
        "task_type": "higher-order ToM",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "low",
        "family_tag": "theory_of_mind",
    },
    "Theory of Mind Timeline": {
        "domain": "social cognition",
        "task_type": "false belief",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "low",
        "family_tag": "theory_of_mind",
    },
    "ToM (Johns Hopkins)": {
        "domain": "social cognition",
        "task_type": "false belief",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "low",
        "family_tag": "theory_of_mind",
    },
    "GPQA": {
        "domain": "knowledge",
        "task_type": "graduate-level science",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "low",
        "family_tag": "knowledge_recall",
    },
    "WinoGrande": {
        "domain": "knowledge",
        "task_type": "commonsense coreference",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "low",
        "family_tag": "knowledge_recall",
    },
    "RACE-h": {
        "domain": "knowledge",
        "task_type": "reading comprehension",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "low",
        "clinical_risk_level": "low",
        "family_tag": "knowledge_recall",
    },
    "MentalChat16K": {
        "domain": "documentation",
        "task_type": "chat quality evaluation",
        "metric_type": "judge score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "documentation",
    },
    "PATIENT-Ψ": {
        "domain": "simulation",
        "task_type": "patient simulation",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "simulation",
    },
    "PATIENT-Ψ-TRAINER": {
        "domain": "simulation",
        "task_type": "training simulation",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "simulation",
    },
    "MusPsy": {
        "domain": "simulation",
        "task_type": "multi-session counseling dataset quality",
        "metric_type": "score",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "multi-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "medium",
        "family_tag": "simulation",
    },
    "MoToMQA": {
        "domain": "social cognition",
        "task_type": "multi-order ToM",
        "metric_type": "accuracy",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "mixed",
        "clinical_risk_level": "low",
        "family_tag": "theory_of_mind",
    },
    "SIRI-2": {
        "domain": "crisis",
        "task_type": "suicide intervention judgment",
        "metric_type": "SIRI-2 score",
        "higher_is_better": False,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "high",
        "clinical_risk_level": "very high",
        "family_tag": "crisis_safety",
    },
    "SIRI-2 reasoning consistency": {
        "domain": "crisis",
        "task_type": "reasoning stability",
        "metric_type": "cosine similarity",
        "higher_is_better": True,
        "single_turn_vs_multi_turn": "single-turn",
        "ambiguity_level": "high",
        "clinical_risk_level": "high",
        "family_tag": "crisis_safety",
    },
}


BENCHMARK_VARIANTS: dict[str, list[dict[str, Any]]] = {
    "MentalBench": [
        {"variant": "Overall", "summary_role": "headline"},
        {"variant": "Type 1", "summary_role": "component"},
        {"variant": "Type 2", "summary_role": "component"},
        {"variant": "Type 3", "summary_role": "component"},
        {"variant": "Type 4", "summary_role": "component"},
    ],
    "MENTAT": [
        {"variant": "Overall Accuracy", "summary_role": "headline"},
        {"variant": "Diagnosis", "summary_role": "component"},
        {"variant": "Treatment", "summary_role": "component"},
        {"variant": "Monitoring", "summary_role": "component"},
        {"variant": "Triage", "summary_role": "component"},
        {"variant": "Documentation", "summary_role": "component"},
    ],
    "PsychiatryBench": [
        {"variant": "Overall", "summary_role": "headline"},
        {"variant": "Specific disorder", "summary_role": "component"},
        {"variant": "Sequential QA", "summary_role": "component"},
        {"variant": "Clinical approach", "summary_role": "component"},
        {"variant": "EMI", "summary_role": "component"},
    ],
    "PsychBench": [
        {"variant": "Overall", "summary_role": "headline"},
    ],
    "PsychoBench": [
        {"variant": "Top-1 Accuracy", "summary_role": "headline", "family_tag": "knowledge_recall"},
        {"variant": "Top-2 Accuracy", "summary_role": "component", "family_tag": "knowledge_recall"},
        {"variant": "F1 (weighted)", "summary_role": "component", "family_tag": "knowledge_recall"},
        {"variant": "Precision (weighted)", "summary_role": "component", "family_tag": "knowledge_recall"},
        {"variant": "Recall (weighted)", "summary_role": "component", "family_tag": "knowledge_recall"},
    ],
    "CBT-Bench": [
        {"variant": "CBT-QA Accuracy", "summary_role": "headline"},
        {"variant": "CBT-CD F1", "summary_role": "component"},
        {"variant": "CBT-PC F1", "summary_role": "component"},
        {"variant": "CBT-FC F1", "summary_role": "component"},
    ],
    "CBT-Bench Level III": [
        {"variant": "Overall vs reference", "summary_role": "headline"},
    ],
    "CounselingBench": [
        {"variant": "Self-consistency", "summary_role": "headline"},
        {"variant": "Zero-shot", "summary_role": "component"},
    ],
    "MindEval": [
        {"variant": "Avg", "summary_role": "headline", "family_tag": "therapy_protocol_fidelity"},
        {"variant": "CAC", "summary_role": "component", "family_tag": "therapy_protocol_fidelity"},
        {"variant": "EPC", "summary_role": "component", "family_tag": "therapy_protocol_fidelity"},
        {"variant": "AR", "summary_role": "component", "family_tag": "therapy_protocol_fidelity"},
        {"variant": "TRA", "summary_role": "component", "family_tag": "therapy_protocol_fidelity"},
        {"variant": "ASCQ", "summary_role": "component", "family_tag": "therapy_protocol_fidelity"},
    ],
    "OMind-Chat": [
        {"variant": "Turn Level Binary", "summary_role": "component", "family_tag": "mental_health_dialogue_quality"},
        {"variant": "Turn Level Scale 0-10", "summary_role": "component", "family_tag": "mental_health_dialogue_quality"},
        {"variant": "Turn Level Likert 1-5", "summary_role": "component", "family_tag": "mental_health_dialogue_quality"},
        {"variant": "Conversation Level Binary", "summary_role": "component", "family_tag": "mental_health_dialogue_quality"},
        {"variant": "Conversation Level Scale 0-10", "summary_role": "headline", "family_tag": "mental_health_dialogue_quality"},
        {"variant": "Conversation Level Likert 1-5", "summary_role": "component", "family_tag": "mental_health_dialogue_quality"},
    ],
    "HEART": [
        {"variant": "H", "summary_role": "component", "family_tag": "multi_turn_consistency"},
        {"variant": "E", "summary_role": "component", "family_tag": "multi_turn_consistency"},
        {"variant": "A", "summary_role": "component", "family_tag": "multi_turn_consistency"},
        {"variant": "R", "summary_role": "component", "family_tag": "multi_turn_consistency"},
        {"variant": "T", "summary_role": "component", "family_tag": "multi_turn_consistency"},
        {"variant": "Overall Win Rate", "summary_role": "headline", "family_tag": "multi_turn_consistency"},
        {"variant": "Elo Rating", "summary_role": "component", "family_tag": "multi_turn_consistency"},
    ],
    "MoralBench": [
        {"variant": "MFQ-30", "summary_role": "component"},
        {"variant": "MFV Total", "summary_role": "headline"},
    ],
    "MoralChoice": [
        {"variant": "Low Ambiguity Accuracy", "summary_role": "headline_low", "family_tag": "moral_reasoning_low_ambiguity"},
        {"variant": "High Ambiguity Accuracy", "summary_role": "headline_high", "family_tag": "moral_reasoning_high_ambiguity"},
    ],
    "EmoBench": [
        {"variant": "Emotion Understanding", "summary_role": "headline", "family_tag": "emotion_understanding"},
        {"variant": "Emotion Application", "summary_role": "headline", "family_tag": "emotion_application"},
    ],
    "EmoBench-Reddit": [
        {"variant": "Overall Accuracy", "summary_role": "headline", "family_tag": "emotion_understanding"},
        {"variant": "Sad Perception 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sad Perception 2(a)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sad Perception 2(b)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sad Perception 3", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sad Cognition 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sad Cognition 2", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sad Cognition 3", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Humor Perception 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Humor Perception 2(a)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Humor Perception 2(b)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Humor Perception 3", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Humor Cognition 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Humor Cognition 2", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Humor Cognition 3", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sarcasm Perception 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sarcasm Perception 2(a)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sarcasm Perception 2(b)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sarcasm Perception 3", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sarcasm Cognition 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sarcasm Cognition 2", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Sarcasm Cognition 3", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Happy Perception 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Happy Perception 2(a)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Happy Perception 2(b)", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Happy Perception 3", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Happy Cognition 1", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Happy Cognition 2", "summary_role": "component", "family_tag": "emotion_understanding"},
        {"variant": "Happy Cognition 3", "summary_role": "component", "family_tag": "emotion_understanding"},
    ],
    "EmoBench-M": [
        {"variant": "FER SOER ACC", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER SOER WAF", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER SPER ACC", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER SPER WAF", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER OSA ACC", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER OSA WAF", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER EIA ACC", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER EIA WAF", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER SEA ACC", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER SEA WAF", "summary_role": "component", "family_tag": "emotion_recognition"},
        {"variant": "FER Avg ACC", "summary_role": "headline", "family_tag": "emotion_recognition"},
        {"variant": "FER Avg WAF", "summary_role": "component", "family_tag": "emotion_recognition"},
    ],
    "MMLU-Pro": [
        {"variant": "Psychology", "summary_role": "headline", "family_tag": "knowledge_recall"},
        {"variant": "Overall", "summary_role": "component", "family_tag": "knowledge_recall"},
    ],
    "Theory of Mind": [
        {"variant": "Unexpected Content", "summary_role": "headline", "family_tag": "theory_of_mind"},
        {"variant": "Transfer DQA", "summary_role": "headline", "family_tag": "theory_of_mind_transfer"},
    ],
    "Theory of Mind Timeline": [
        {"variant": "False Belief", "summary_role": "headline"},
    ],
    "ToM (Johns Hopkins)": [
        {"variant": "Overall", "summary_role": "headline"},
    ],
    "GPQA": [
        {"variant": "Overall", "summary_role": "headline"},
    ],
    "WinoGrande": [
        {"variant": "Overall", "summary_role": "headline"},
    ],
    "RACE-h": [
        {"variant": "Overall", "summary_role": "headline"},
    ],
    "MentalChat16K": [
        {"variant": "Judge average 1", "summary_role": "headline"},
        {"variant": "Judge average 2", "summary_role": "component"},
    ],
    "PATIENT-Ψ": [
        {"variant": "Expert pairwise μ", "summary_role": "headline"},
        {"variant": "Confidence μ", "summary_role": "component"},
    ],
    "PATIENT-Ψ-TRAINER": [
        {"variant": "Expert μ", "summary_role": "headline"},
        {"variant": "Trainee μ", "summary_role": "component"},
    ],
    "MusPsy": [
        {"variant": "GPT-4o eval average", "summary_role": "headline"},
        {"variant": "Human eval average", "summary_role": "component"},
    ],
    "MoToMQA": [
        {"variant": "Aggregate ToM Score", "summary_role": "headline"},
        {"variant": "Factual Accuracy", "summary_role": "component"},
        {"variant": "ToM 6th-Order Task", "summary_role": "component"},
    ],
    "SIRI-2": [
        {"variant": "Configuration mean", "summary_role": "headline"},
    ],
    "SIRI-2 reasoning consistency": [
        {"variant": "Mean consistency", "summary_role": "headline", "family_tag": "crisis_safety"},
    ],
}


def build_benchmark_map() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, variants in BENCHMARK_VARIANTS.items():
        meta = BENCHMARK_META[benchmark]
        for variant in variants:
            rows.append({
                "benchmark": benchmark,
                "benchmark_variant": variant["variant"],
                "domain": meta["domain"],
                "task_type": meta["task_type"],
                "metric_type": meta["metric_type"],
                "higher_is_better": meta["higher_is_better"],
                "single_turn_vs_multi_turn": meta["single_turn_vs_multi_turn"],
                "ambiguity_level": meta["ambiguity_level"],
                "clinical_risk_level": meta["clinical_risk_level"],
                "family_tag": variant.get("family_tag", meta["family_tag"]),
                "summary_role": variant["summary_role"],
                "notes": variant.get("notes", ""),
            })
    return rows


def build_model_metadata(model_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in model_rows:
        out[row["model_name_norm"]] = row
    out.setdefault("Claude Opus 4", {
        "model_name_norm": "Claude Opus 4",
        "model_family": "Claude Opus 4",
        "provider": "Anthropic",
        "model_type": "closed",
        "release_date": "2025-05-22",
        "release_year": "2025",
        "release_month": "5",
        "release_month_index": "24305",
        "release_month_key": "2025-05",
        "aa_model_name": "Claude 4 Opus (Non-reasoning)",
        "aa_intelligence_index": "33",
        "aa_source": "https://artificialanalysis.ai/models/claude-4-opus",
    })
    return out


def infer_provider(name: str) -> str:
    name_l = norm(name)
    if any(tok in name_l for tok in ["gpt", "chatgpt", "o1", "o3", "openai"]):
        return "OpenAI"
    if "claude" in name_l or "anthropic" in name_l or "sonnet" in name_l or "haiku" in name_l:
        return "Anthropic"
    if "gemini" in name_l or "gemma" in name_l or "medgemma" in name_l or "palm" in name_l:
        return "Google"
    if "deepseek" in name_l:
        return "DeepSeek"
    if "qwen" in name_l or "qwq" in name_l or "qwq" in name_l:
        return "Alibaba"
    if "grok" in name_l:
        return "xAI"
    if "llama" in name_l or "meta" in name_l:
        return "Meta"
    return ""


def infer_benchmark_variant(row: dict[str, Any]) -> str:
    if row.get("appendix_section") == "A.3":
        return "Configuration mean"
    if row.get("appendix_section") == "A.4":
        return "Mean consistency"
    variant = row.get("dimension") or row.get("metric") or row.get("task") or row.get("benchmark_or_framework")
    return str(variant).strip()


def build_summary_rows(
    benchmark_rows: list[dict[str, Any]],
    supplemental_rows: list[dict[str, Any]],
    model_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    map_lookup = {(key(r["benchmark"]), key(r["benchmark_variant"])): r for r in build_benchmark_map()}

    def add_row(source: dict[str, Any], benchmark: str, variant: str, entity_label: str, entity_type: str, value: float, score_text: str | None = None) -> None:
        meta = map_lookup.get((key(benchmark), key(variant)), {})
        model_info = model_meta.get(entity_label, {})
        provider = model_info.get("provider", "") or (infer_provider(entity_label) if entity_type == "model" else "")
        rows.append({
            "benchmark": benchmark,
            "benchmark_variant": variant,
            "summary_role": meta.get("summary_role", "headline"),
            "domain": meta.get("domain", ""),
            "task_type": meta.get("task_type", ""),
            "metric_type": meta.get("metric_type", ""),
            "higher_is_better": meta.get("higher_is_better", True),
            "single_turn_vs_multi_turn": meta.get("single_turn_vs_multi_turn", ""),
            "ambiguity_level": meta.get("ambiguity_level", ""),
            "clinical_risk_level": meta.get("clinical_risk_level", ""),
            "family_tag": meta.get("family_tag", ""),
            "entity_type": entity_type,
            "entity_label": entity_label,
            "model_name_norm": entity_label if entity_type == "model" else "",
            "provider": provider,
            "model_family": model_info.get("model_family", "") if entity_type == "model" else "",
            "model_type": model_info.get("model_type", "") if entity_type == "model" else "",
            "parameter_size_billions": model_info.get("parameter_size_billions", "") if entity_type == "model" else "",
            "release_date": model_info.get("release_date", "") if entity_type == "model" else "",
            "aa_model_name": model_info.get("aa_model_name", "") if entity_type == "model" else "",
            "aa_intelligence_index": model_info.get("aa_intelligence_index", "") if entity_type == "model" else "",
            "healthbench_overall": model_info.get("healthbench_overall", "") if entity_type == "model" else "",
            "healthbench_consensus_overall": model_info.get("healthbench_consensus_overall", "") if entity_type == "model" else "",
            "source_type": source.get("source_type", ""),
            "source_label": source.get("source_label", ""),
            "source_title": source.get("source_title", ""),
            "source_url": source.get("source_url", ""),
            "score_value": value,
            "score_text": score_text if score_text is not None else str(value),
            "benchmark_group": source.get("benchmark_group", ""),
            "benchmark_source": source.get("benchmark", ""),
            "dimension": source.get("dimension", ""),
            "metric": source.get("metric", ""),
            "rank": source.get("rank", ""),
            "prompt_variant": source.get("prompt_variant", ""),
            "temperature": source.get("temperature", ""),
            "mae_per_item": source.get("mae_per_item", ""),
            "stat_sd": source.get("stat_sd", ""),
            "stat_min_item": source.get("stat_min_item", ""),
            "stat_r": source.get("stat_r", ""),
            "stat_p": source.get("stat_p", ""),
            "stat_n": source.get("stat_n", ""),
            "notes": source.get("notes", ""),
        })

    for row in benchmark_rows:
        benchmark = row["benchmark"]
        variant = infer_benchmark_variant(row)
        value = parse_float(row.get("score_value")) or parse_float(row.get("value")) or parse_float(row.get("score_text"))
        if value is None:
            continue
        model_name = row.get("model_name_norm") or row.get("model_or_subject") or ""
        entity_type = "model" if model_name in model_meta else "reference"
        add_row(
            source=row,
            benchmark=benchmark,
            variant=variant,
            entity_label=model_name,
            entity_type=entity_type,
            value=value,
            score_text=row.get("score_text") or row.get("value") or "",
        )

    # Add SIRI-2 appendix aggregates as model-level rows.
    siri_config_rows = [r for r in supplemental_rows if r.get("appendix_section") == "A.3" and r.get("analysis_kind") == "configuration_score"]
    config_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in siri_config_rows:
        config_groups[row["model_or_subject"]].append(row)
    for model, group in config_groups.items():
        scores = [parse_float(r.get("value")) for r in group]
        scores = [s for s in scores if s is not None]
        maes = [parse_float(r.get("mae_per_item")) for r in group]
        maes = [m for m in maes if m is not None]
        if not scores:
            continue
        add_row(
            source=group[0],
            benchmark="SIRI-2",
            variant="Configuration mean",
            entity_label=model,
            entity_type="model",
            value=mean(scores),
            score_text=f"{mean(scores):.2f}",
        )
        rows[-1]["score_min"] = min(scores)
        rows[-1]["score_max"] = max(scores)
        rows[-1]["score_range"] = max(scores) - min(scores)
        rows[-1]["mae_per_item"] = mean(maes) if maes else ""
        rows[-1]["summary_role"] = "headline"

    siri_consistency_rows = [r for r in supplemental_rows if r.get("appendix_section") == "A.4" and r.get("analysis_kind") == "reasoning_consistency_summary"]
    consistency_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in siri_consistency_rows:
        consistency_groups[row["model_or_subject"]].append(row)
    for model, group in consistency_groups.items():
        values = [parse_float(r.get("value")) for r in group]
        values = [v for v in values if v is not None]
        if not values:
            continue
        meta_row = group[0]
        add_row(
            source=meta_row,
            benchmark="SIRI-2 reasoning consistency",
            variant="Mean consistency",
            entity_label=model,
            entity_type="model",
            value=mean(values),
            score_text=f"{mean(values):.3f}",
        )

    # Add MoToMQA and any other supplemental benchmark-like numeric rows.
    extra_benchmarks = {"MoToMQA", "MentalBench-100k"}
    for row in supplemental_rows:
        benchmark = row.get("benchmark_or_framework") or ""
        if benchmark not in extra_benchmarks:
            continue
        value = parse_float(row.get("value"))
        if value is None:
            continue
        variant = row.get("metric") or row.get("dimension") or "Overall"
        entity_label = row.get("model_or_subject") or row.get("model_name_norm") or ""
        add_row(
            source=row,
            benchmark=benchmark,
            variant=variant,
            entity_label=entity_label,
            entity_type="reference" if entity_label not in model_meta or "human" in norm(entity_label) else "model",
            value=value,
            score_text=row.get("score_text") or row.get("value") or "",
        )

    return rows


def normalize_group_percentiles(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("score_value") in ("", None):
            continue
        grouped[(row.get("benchmark", ""), row.get("benchmark_variant", ""))].append(row)
    for _, group in grouped.items():
        if not group:
            continue
        higher = str(group[0].get("higher_is_better", True)).lower() in {"true", "1", "yes"}
        group.sort(key=lambda row: float(row["score_value"]), reverse=higher)
        n = len(group)
        for idx, row in enumerate(group):
            if n == 1:
                row["normalized_pct"] = 100.0
            else:
                row["normalized_pct"] = round(100.0 * (n - idx - 1) / (n - 1), 2)


def raw_ability_score_pct(row: dict[str, Any]) -> float | None:
    value = parse_float(row.get("score_value"))
    if value is None:
        return None
    benchmark = str(row.get("benchmark", ""))
    variant = str(row.get("benchmark_variant", ""))
    if benchmark == "SIRI-2":
        return max(0.0, min(100.0, 100.0 - value))
    if benchmark == "CBT-Bench Level III" and variant == "Overall vs reference":
        return max(0.0, min(100.0, (value + 2.0) * 25.0))
    if benchmark == "OMind-Chat" and "Scale 0-10" in variant:
        return max(0.0, min(100.0, value * 10.0))
    if benchmark == "MentalChat16K":
        return max(0.0, min(100.0, value * 10.0))
    if benchmark == "MindEval":
        # MindEval uses a 1-6 scoring rubric (Sword Health, 2026).
        return max(0.0, min(100.0, (value - 1.0) / 5.0 * 100.0))
    if benchmark == "MusPsy":
        # MusPsy evaluation uses a 0-5 Likert scale.
        return max(0.0, min(100.0, value * 20.0))
    if 0.0 <= value <= 1.0:
        return max(0.0, min(100.0, value * 100.0))
    if 0.0 <= value <= 10.0 and "0-10" in variant:
        return max(0.0, min(100.0, value * 10.0))
    if 0.0 <= value <= 100.0:
        return value
    return None


def attach_raw_ability_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        score = raw_ability_score_pct(row)
        row["ability_score_pct"] = round(score, 2) if score is not None else ""


def build_jagged_profile(summary_rows: list[dict[str, Any]], model_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    family_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row.get("entity_type") != "model":
            continue
        if not str(row.get("summary_role", "")).startswith("headline"):
            continue
        family = row.get("family_tag") or ""
        if not family:
            continue
        family_rows[(family, row["entity_label"])].append(row)
    out: list[dict[str, Any]] = []
    for (family, model), rows in family_rows.items():
        pcts = [parse_float(r.get("ability_score_pct")) for r in rows]
        pcts = [p for p in pcts if p is not None]
        if not pcts:
            continue
        meta = model_meta.get(model, {})
        provider = meta.get("provider", "") or infer_provider(model)
        benchmark_list = sorted({f"{r['benchmark']} ({r['benchmark_variant']})" for r in rows})
        out.append({
            "capability_family": family,
            "model_name_norm": model,
            "provider": provider,
            "model_family": meta.get("model_family", ""),
            "model_type": meta.get("model_type", ""),
            "parameter_size_billions": meta.get("parameter_size_billions", ""),
            "release_date": meta.get("release_date", ""),
            **release_cohort_fields(meta.get("release_date", "")),
            "aa_intelligence_index": meta.get("aa_intelligence_index", ""),
            "healthbench_overall": meta.get("healthbench_overall", ""),
            "healthbench_consensus_overall": meta.get("healthbench_consensus_overall", ""),
            "healthbench_best_available": meta.get("healthbench_overall", "") or meta.get("healthbench_consensus_overall", ""),
            "family_score_pct": round(mean(pcts), 2),
            "family_score_min_pct": round(min(pcts), 2),
            "family_score_max_pct": round(max(pcts), 2),
            "benchmark_count": len(benchmark_list),
            "benchmark_list": "; ".join(benchmark_list),
        })
    out.sort(key=lambda r: (r["capability_family"], -(parse_float(r.get("family_score_pct")) or 0), r["model_name_norm"]))
    return out


def build_frontier_reference(frontier_rows: list[dict[str, Any]], model_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in frontier_rows:
        model = row.get("model_name_norm", "")
        meta = model_meta.get(model, {})
        provider = meta.get("provider", "") or infer_provider(model)
        notes = row.get("notes", "")
        name_l = norm(model)
        notes_l = norm(notes)
        if any(tok in name_l for tok in ["o1", "o3", "deepseek-r1", "thinking"]) or "extended-thinking" in notes_l:
            thinking_mode = "thinking"
        elif any(tok in name_l for tok in ["mini", "flash", "haiku", "turbo"]):
            thinking_mode = "non-thinking"
        else:
            thinking_mode = "unknown"
        out.append({
            "model_name_norm": model,
            "family": meta.get("model_family", "") or model,
            "provider": row.get("provider", "") or provider,
            "model_type": row.get("model_type", ""),
            "parameter_size_billions": row.get("parameter_size_billions", ""),
            "release_date": row.get("release_date", ""),
            **release_cohort_fields(row.get("release_date", "")),
            "benchmark_date": "",
            "benchmark_date_note": "",
            "thinking_mode": thinking_mode,
            "thinking_mode_note": "heuristic from name/notes",
            "open_closed": row.get("model_type", ""),
            "aa_intelligence_index": row.get("aa_intelligence_index", ""),
            "aa_source": row.get("aa_source", ""),
            "healthbench_overall": row.get("healthbench_overall", ""),
            "healthbench_consensus_overall": row.get("healthbench_consensus_overall", ""),
            "healthbench_source": row.get("healthbench_source", ""),
            "notes": row.get("notes", ""),
        })
    return out


def build_frontier_timeline(frontier_reference_rows: list[dict[str, Any]], model_signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signal_lookup = {row.get("model_name_norm", ""): row for row in model_signal_rows}
    out: list[dict[str, Any]] = []
    for row in frontier_reference_rows:
        model = row.get("model_name_norm", "")
        signal = signal_lookup.get(model, {})
        release_date = row.get("release_date", "") or signal.get("release_date", "")
        cohort = release_cohort_fields(release_date)
        out.append({
            "model_name_norm": model,
            "family": row.get("family", ""),
            "provider": row.get("provider", ""),
            "model_type": row.get("model_type", ""),
            "parameter_size_billions": row.get("parameter_size_billions", "") or signal.get("parameter_size_billions", ""),
            "release_date": release_date,
            "release_year": cohort["release_year"],
            "release_month": cohort["release_month"],
            "release_month_index": cohort["release_month_index"],
            "release_month_key": cohort["release_month_key"],
            "thinking_mode": row.get("thinking_mode", ""),
            "open_closed": row.get("open_closed", ""),
            "aa_intelligence_index": row.get("aa_intelligence_index", "") or signal.get("aa_intelligence_index", ""),
            "healthbench_overall": row.get("healthbench_overall", "") or signal.get("healthbench_overall", ""),
            "healthbench_consensus_overall": row.get("healthbench_consensus_overall", "") or signal.get("healthbench_consensus_overall", ""),
            "benchmark_count": signal.get("benchmark_count", ""),
            "overall_psych_score_pct": signal.get("overall_psych_score_pct", ""),
            "jaggedness_score_pct": signal.get("jaggedness_score_pct", ""),
            "stability_score_pct": signal.get("stability_score_pct", ""),
            "psych_balance_score_pct": signal.get("psych_balance_score_pct", ""),
        })
    out.sort(key=lambda r: (parse_int(r.get("release_month_index")) or 999999, r.get("provider", ""), r.get("model_name_norm", "")))
    return out


def build_frontier_monthly_summary(frontier_timeline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frontier_timeline_rows:
        key = row.get("release_month_key", "")
        if key:
            grouped[key].append(row)

    def mean_of(rows: list[dict[str, Any]], field: str) -> float | None:
        vals = [parse_float(r.get(field)) for r in rows if parse_float(r.get(field)) is not None]
        if not vals:
            return None
        return round(float(mean(vals)), 2)

    out: list[dict[str, Any]] = []
    for month_key, rows in sorted(grouped.items()):
        models = sorted({r.get("model_name_norm", "") for r in rows if r.get("model_name_norm")})
        release_year = parse_int(month_key[:4]) if len(month_key) >= 4 else None
        release_month = parse_int(month_key[5:7]) if len(month_key) >= 7 else None
        release_month_index = parse_int(rows[0].get("release_month_index"))
        out.append({
            "release_month_key": month_key,
            "release_year": release_year if release_year is not None else "",
            "release_month": release_month if release_month is not None else "",
            "release_month_index": release_month_index if release_month_index is not None else "",
            "frontier_model_count": len(rows),
            "model_list": "; ".join(models),
            "mean_parameter_size_billions": mean_of(rows, "parameter_size_billions"),
            "mean_aa_intelligence_index": mean_of(rows, "aa_intelligence_index"),
            "mean_overall_psych_score_pct": mean_of(rows, "overall_psych_score_pct"),
            "mean_jaggedness_score_pct": mean_of(rows, "jaggedness_score_pct"),
            "mean_stability_score_pct": mean_of(rows, "stability_score_pct"),
            "mean_psych_balance_score_pct": mean_of(rows, "psych_balance_score_pct"),
        })
    return out


def dedupe_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(
            row.get("benchmark", ""),
            row.get("benchmark_variant", ""),
            row.get("entity_type", ""),
            row.get("entity_label", ""),
        )].append(row)

    numeric_fields = [
        "score_value",
        "score_min",
        "score_max",
        "score_range",
        "mae_per_item",
        "stat_sd",
        "stat_min_item",
        "stat_r",
        "stat_p",
        "stat_n",
    ]
    merged_fields = [
        "source_type",
        "source_label",
        "source_title",
        "source_url",
        "benchmark_group",
        "benchmark_source",
        "dimension",
        "metric",
        "rank",
        "prompt_variant",
        "temperature",
        "notes",
    ]

    out: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        base = dict(rows[0])
        for field in numeric_fields:
            values = [parse_float(r.get(field)) for r in rows]
            values = [v for v in values if v is not None]
            if values:
                base[field] = round(mean(values), 6)
        for field in merged_fields:
            values: list[str] = []
            for row in rows:
                value = str(row.get(field, "") or "").strip()
                if value and value not in values:
                    values.append(value)
            base[field] = "; ".join(values)
        raw_values = [parse_float(r.get("score_value")) for r in rows]
        raw_values = [v for v in raw_values if v is not None]
        if raw_values:
            base["score_value"] = round(mean(raw_values), 6)
            base["score_value_min_observed"] = round(min(raw_values), 6)
            base["score_value_max_observed"] = round(max(raw_values), 6)
        base["source_count"] = len(rows)
        base["duplicate_row_count"] = len(rows) - 1
        out.append(base)
    return out


def rows_starting_with_headline(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        if row.get("entity_type") != "model":
            continue
        if not str(row.get("summary_role", "")).startswith("headline"):
            continue
        out.append(row)
    return out


def attach_model_meta(row: dict[str, Any], model_name: str, model_meta: dict[str, dict[str, Any]]) -> None:
    meta = model_meta.get(model_name, {})
    row["provider"] = meta.get("provider", "") or infer_provider(model_name)
    row["model_family"] = meta.get("model_family", "")
    row["model_type"] = meta.get("model_type", "")
    row["parameter_size_billions"] = meta.get("parameter_size_billions", "")
    row["release_date"] = meta.get("release_date", "")
    row.update(release_cohort_fields(row["release_date"]))
    row["aa_intelligence_index"] = meta.get("aa_intelligence_index", "")
    row["healthbench_overall"] = meta.get("healthbench_overall", "")
    row["healthbench_consensus_overall"] = meta.get("healthbench_consensus_overall", "")
    row["healthbench_best_available"] = meta.get("healthbench_overall", "") or meta.get("healthbench_consensus_overall", "")


def build_benchmark_composite(summary_rows: list[dict[str, Any]], model_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_starting_with_headline(summary_rows):
        grouped[(row["entity_label"], row["benchmark"])].append(row)

    out: list[dict[str, Any]] = []
    for (model_name, benchmark), rows in grouped.items():
        pcts = [parse_float(r.get("ability_score_pct")) for r in rows]
        pcts = [p for p in pcts if p is not None]
        raw_scores = [parse_float(r.get("score_value")) for r in rows]
        raw_scores = [v for v in raw_scores if v is not None]
        if not pcts:
            continue
        row = {
            "model_name_norm": model_name,
            "benchmark": benchmark,
            "domain": rows[0].get("domain", ""),
            "task_type": rows[0].get("task_type", ""),
            "single_turn_vs_multi_turn": rows[0].get("single_turn_vs_multi_turn", ""),
            "ambiguity_level": rows[0].get("ambiguity_level", ""),
            "clinical_risk_level": rows[0].get("clinical_risk_level", ""),
            "benchmark_score_pct": round(mean(pcts), 2),
            "benchmark_score_pct_min": round(min(pcts), 2),
            "benchmark_score_pct_max": round(max(pcts), 2),
            "benchmark_score_raw_mean": round(mean(raw_scores), 6) if raw_scores else "",
            "benchmark_variant_count": len(rows),
            "benchmark_variant_list": "; ".join(sorted({str(r.get('benchmark_variant', '')) for r in rows})),
            "family_tag_list": "; ".join(sorted({str(r.get('family_tag', '')) for r in rows if str(r.get('family_tag', '')).strip()})),
        }
        attach_model_meta(row, model_name, model_meta)
        out.append(row)
    out.sort(key=lambda r: (r["benchmark"], -(parse_float(r.get("benchmark_score_pct")) or 0), r["model_name_norm"]))
    return out


def build_model_psych_composite(
    benchmark_composite_rows: list[dict[str, Any]],
    jagged_rows: list[dict[str, Any]],
    model_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    benchmark_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_composite_rows:
        benchmark_grouped[row["model_name_norm"]].append(row)

    family_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jagged_rows:
        family_grouped[row["model_name_norm"]].append(row)

    out: list[dict[str, Any]] = []
    model_names = sorted(set(benchmark_grouped) | set(family_grouped))
    for model_name in model_names:
        benchmark_rows = benchmark_grouped.get(model_name, [])
        family_rows = family_grouped.get(model_name, [])
        family_scores = [parse_float(r.get("family_score_pct")) for r in family_rows]
        family_scores = [score for score in family_scores if score is not None]
        if not family_scores:
            continue
        benchmark_names = sorted({str(r.get("benchmark", "")) for r in benchmark_rows if str(r.get("benchmark", "")).strip()})
        family_names = sorted({str(r.get("capability_family", "")) for r in family_rows if str(r.get("capability_family", "")).strip()})
        benchmark_count = len(benchmark_names)
        family_count = len(family_names)
        row = {
            "model_name_norm": model_name,
            "overall_psych_score_pct": round(mean(family_scores), 2),
            "overall_psych_score_pct_min": round(min(family_scores), 2),
            "overall_psych_score_pct_max": round(max(family_scores), 2),
            "psych_score_variability_sd": round(pstdev(family_scores), 2) if len(family_scores) > 1 else 0.0,
            "score_definition": "family_balanced_mean_of_family_means",
            "benchmark_count": benchmark_count,
            "benchmark_list": "; ".join(benchmark_names),
            "family_count": family_count,
            "capability_family_list": "; ".join(family_names),
            "coverage_tier": coverage_tier(benchmark_count),
            "main_visual_eligible": "yes" if benchmark_count >= 4 else "",
            "complete_profile_eligible": "yes" if benchmark_count >= 7 else "",
        }
        attach_model_meta(row, model_name, model_meta)
        out.append(row)
    out.sort(key=lambda r: (-(parse_float(r.get("overall_psych_score_pct")) or 0), -(parse_int(r.get("benchmark_count")) or 0), r["model_name_norm"]))
    return out


def build_model_signal_composite(
    model_psych_rows: list[dict[str, Any]],
    jagged_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_df = pd.DataFrame(model_psych_rows)
    jagged_df = pd.DataFrame(jagged_rows)
    gap_df = pd.DataFrame(gap_rows)

    model_df["overall_psych_score_pct"] = pd.to_numeric(model_df["overall_psych_score_pct"], errors="coerce")
    model_df["benchmark_count"] = pd.to_numeric(model_df["benchmark_count"], errors="coerce")
    model_df["family_count"] = pd.to_numeric(model_df["family_count"], errors="coerce")
    jagged_df["family_score_pct"] = pd.to_numeric(jagged_df["family_score_pct"], errors="coerce")
    gap_df["gap_score_pct"] = pd.to_numeric(gap_df["gap_score_pct"], errors="coerce")

    fam_pivot = jagged_df.pivot_table(index="model_name_norm", columns="capability_family", values="family_score_pct", aggfunc="mean")
    family_range = pd.DataFrame({
        "model_name_norm": fam_pivot.index,
        "family_score_min_pct": fam_pivot.min(axis=1, skipna=True).values,
        "family_score_max_pct": fam_pivot.max(axis=1, skipna=True).values,
    })
    family_range["family_score_range_pct"] = family_range["family_score_max_pct"] - family_range["family_score_min_pct"]

    gap_abs = gap_df.copy()
    gap_abs["abs_gap_score_pct"] = gap_abs["gap_score_pct"].abs()
    gap_abs = gap_abs.groupby("model_name_norm", as_index=False)["abs_gap_score_pct"].mean().rename(columns={"abs_gap_score_pct": "gap_abs_mean_pct"})

    out = model_df.merge(family_range[["model_name_norm", "family_score_range_pct"]], on="model_name_norm", how="left")
    out = out.merge(gap_abs, on="model_name_norm", how="left")

    out["overall_psych_percentile_pct"] = out["overall_psych_score_pct"].rank(method="average", pct=True) * 100.0
    out["jaggedness_eligible"] = (out["benchmark_count"] >= 4) & (out["family_count"] >= 2)
    out["family_range_percentile_pct"] = float("nan")
    out["gap_abs_percentile_pct"] = float("nan")
    eligible_family = out.loc[out["jaggedness_eligible"] & out["family_score_range_pct"].notna(), "family_score_range_pct"]
    if not eligible_family.empty:
        out.loc[eligible_family.index, "family_range_percentile_pct"] = eligible_family.rank(method="average", pct=True) * 100.0
    eligible_gap = out.loc[out["jaggedness_eligible"] & out["gap_abs_mean_pct"].notna(), "gap_abs_mean_pct"]
    if not eligible_gap.empty:
        out.loc[eligible_gap.index, "gap_abs_percentile_pct"] = eligible_gap.rank(method="average", pct=True) * 100.0
    out["family_range_inverse_percentile_pct"] = 100.0 - out["family_range_percentile_pct"]
    out["gap_abs_inverse_percentile_pct"] = 100.0 - out["gap_abs_percentile_pct"]

    def mean_available(row: Any, fields: list[str]) -> float | None:
        vals = [parse_float(row.get(field)) for field in fields if parse_float(row.get(field)) is not None]
        if not vals:
            return None
        return float(mean(vals))

    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        jaggedness = None
        stability = None
        if bool(row.get("jaggedness_eligible")):
            jaggedness = mean_available(row, [
                "family_range_percentile_pct",
                "gap_abs_percentile_pct",
            ])
            stability = mean_available(row, [
                "family_range_inverse_percentile_pct",
                "gap_abs_inverse_percentile_pct",
            ])
        psych_strength = parse_float(row.get("overall_psych_percentile_pct"))
        release_year = parse_int(str(row.get("release_date", ""))[:4]) if str(row.get("release_date", "")).strip() else None
        cohort = release_cohort_fields(row.get("release_date"))
        params = parse_float(row.get("parameter_size_billions"))
        log_params = math.log10(params) if params and params > 0 else None
        if psych_strength is None or stability is None:
            psych_balance = None
        else:
            psych_balance = round(mean([psych_strength, stability]), 2)
        rows.append({
            "model_name_norm": row.get("model_name_norm", ""),
            "overall_psych_score_pct": row.get("overall_psych_score_pct", ""),
            "overall_psych_percentile_pct": round(psych_strength, 2) if psych_strength is not None else "",
            "family_score_range_pct": row.get("family_score_range_pct", ""),
            "gap_abs_mean_pct": row.get("gap_abs_mean_pct", ""),
            "family_range_percentile_pct": round(parse_float(row.get("family_range_percentile_pct")) or 0, 2) if parse_float(row.get("family_range_percentile_pct")) is not None else "",
            "gap_abs_percentile_pct": round(parse_float(row.get("gap_abs_percentile_pct")) or 0, 2) if parse_float(row.get("gap_abs_percentile_pct")) is not None else "",
            "jaggedness_score_pct": round(jaggedness, 2) if jaggedness is not None else "",
            "stability_score_pct": round(stability, 2) if stability is not None else "",
            "psych_balance_score_pct": psych_balance if psych_balance is not None else "",
            "benchmark_count": row.get("benchmark_count", ""),
            "family_count": row.get("family_count", ""),
            "score_definition": row.get("score_definition", ""),
            "coverage_tier": row.get("coverage_tier", ""),
            "main_visual_eligible": row.get("main_visual_eligible", ""),
            "complete_profile_eligible": row.get("complete_profile_eligible", ""),
            "jaggedness_eligible": "yes" if bool(row.get("jaggedness_eligible")) else "",
            "provider": row.get("provider", ""),
            "model_family": row.get("model_family", ""),
            "model_type": row.get("model_type", ""),
            "parameter_size_billions": row.get("parameter_size_billions", ""),
            "release_date": row.get("release_date", ""),
            "release_year": release_year if release_year is not None else "",
            "release_month": cohort["release_month"],
            "release_month_index": cohort["release_month_index"],
            "release_month_key": cohort["release_month_key"],
            "log_params": round(log_params, 4) if log_params is not None else "",
            "aa_intelligence_index": row.get("aa_intelligence_index", ""),
            "healthbench_overall": row.get("healthbench_overall", ""),
            "healthbench_consensus_overall": row.get("healthbench_consensus_overall", ""),
            "healthbench_best_available": row.get("healthbench_best_available", ""),
        })

    rows.sort(key=lambda r: (-(parse_float(r.get("psych_balance_score_pct")) or 0), r["model_name_norm"]))
    return rows


def correlation_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if len(pairs) < 3:
        return {
            "n": len(pairs),
            "pearson_r": "",
            "pearson_p": "",
            "spearman_rho": "",
            "spearman_p": "",
        }
    xs = pd.Series([p[0] for p in pairs], dtype="float64")
    ys = pd.Series([p[1] for p in pairs], dtype="float64")
    if xs.nunique(dropna=True) < 2 or ys.nunique(dropna=True) < 2:
        return {
            "n": len(pairs),
            "pearson_r": "",
            "pearson_p": "",
            "spearman_rho": "",
            "spearman_p": "",
        }
    pearson_val = xs.corr(ys, method="pearson")
    spearman_val = xs.rank(method="average").corr(ys.rank(method="average"), method="pearson")

    def approx_p(value: float | None, n: int) -> float | None:
        if value is None or n < 4 or abs(value) >= 1:
            return None
        z = abs(0.5 * math.log((1 + value) / (1 - value))) * math.sqrt(max(n - 3, 1))
        return round(2 * (1 - NormalDist().cdf(z)), 6)

    return {
        "n": len(pairs),
        "pearson_r": round(float(pearson_val), 4) if pearson_val is not None and not math.isnan(float(pearson_val)) else "",
        "pearson_p": approx_p(float(pearson_val), len(pairs)) if pearson_val is not None and not math.isnan(float(pearson_val)) else "",
        "spearman_rho": round(float(spearman_val), 4) if spearman_val is not None and not math.isnan(float(spearman_val)) else "",
        "spearman_p": approx_p(float(spearman_val), len(pairs)) if spearman_val is not None and not math.isnan(float(spearman_val)) else "",
    }


def build_correlation_rows(
    model_psych_rows: list[dict[str, Any]],
    jagged_rows: list[dict[str, Any]],
    model_signal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def collect_pairs(rows: list[dict[str, Any]], x_field: str, y_field: str) -> list[tuple[float, float]]:
        pairs: list[tuple[float, float]] = []
        for row in rows:
            x = parse_float(row.get(x_field))
            y = parse_float(row.get(y_field))
            if x is None or y is None:
                continue
            pairs.append((x, y))
        return pairs

    out: list[dict[str, Any]] = []
    signal_scopes = [
        ("overall_all_models", "All psych models", model_signal_rows),
        (
            "overall_multi_benchmark",
            "All psych models (benchmark_count >= 2)",
            [row for row in model_signal_rows if (parse_int(row.get("benchmark_count")) or 0) >= 2],
        ),
        (
            "overall_profile_4plus",
            "All psych models (benchmark_count >= 4)",
            [row for row in model_signal_rows if (parse_int(row.get("benchmark_count")) or 0) >= 4],
        ),
        (
            "frontier_2025_2026",
            "Frontier models released 2025-2026",
            [row for row in model_signal_rows if (parse_int(row.get("release_year")) or 0) >= 2025],
        ),
        (
            "frontier_2025_2026_multi_benchmark",
            "Frontier models released 2025-2026 (benchmark_count >= 2)",
            [
                row for row in model_signal_rows
                if (parse_int(row.get("release_year")) or 0) >= 2025 and (parse_int(row.get("benchmark_count")) or 0) >= 2
            ],
        ),
        (
            "frontier_2025_2026_profile_4plus",
            "Frontier models released 2025-2026 (benchmark_count >= 4)",
            [
                row for row in model_signal_rows
                if (parse_int(row.get("release_year")) or 0) >= 2025 and (parse_int(row.get("benchmark_count")) or 0) >= 4
            ],
        ),
    ]
    external_specs = [
        ("aa_intelligence_index", "AA Intelligence Index"),
        ("healthbench_overall", "HealthBench Overall"),
        ("healthbench_consensus_overall", "HealthBench Consensus"),
        ("healthbench_best_available", "HealthBench Best Available"),
        ("log_params", "Log10(Parameter Size)"),
        ("parameter_size_billions", "Parameter Size (B)"),
        ("release_year", "Release Year"),
        ("release_month_index", "Release Month Index"),
    ]
    psych_metrics = [
        "overall_psych_score_pct",
        "jaggedness_score_pct",
        "stability_score_pct",
        "psych_balance_score_pct",
    ]
    for scope_name, scope_label, scoped_rows in signal_scopes:
        for psych_metric in psych_metrics:
            for external_field, external_label in external_specs:
                stats = correlation_stats(collect_pairs(scoped_rows, psych_metric, external_field))
                out.append({
                    "analysis_scope": scope_name,
                    "scope_label": scope_label,
                    "psych_metric": psych_metric,
                    "external_metric": external_field,
                    "external_metric_label": external_label,
                    **stats,
                })

    jagged_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jagged_rows:
        jagged_groups[row["capability_family"]].append(row)
    for family, rows in sorted(jagged_groups.items()):
        for external_field, external_label in external_specs:
            stats = correlation_stats(collect_pairs(rows, "family_score_pct", external_field))
            out.append({
                "analysis_scope": "family",
                "scope_label": family,
                "psych_metric": "family_score_pct",
                "external_metric": external_field,
                "external_metric_label": external_label,
                **stats,
            })
    return out


def build_gap_rows(
    summary_rows: list[dict[str, Any]],
    benchmark_composite_rows: list[dict[str, Any]],
    jagged_rows: list[dict[str, Any]],
    model_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    headline_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows_starting_with_headline(summary_rows):
        headline_lookup[(row["entity_label"], row["benchmark"], row["benchmark_variant"])] = row

    benchmark_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in benchmark_composite_rows:
        benchmark_lookup[(row["model_name_norm"], row["benchmark"])] = row

    family_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in jagged_rows:
        family_lookup[(row["model_name_norm"], row["capability_family"])] = row

    models = sorted({row["entity_label"] for row in rows_starting_with_headline(summary_rows)})
    out: list[dict[str, Any]] = []

    def add_gap(
        model_name: str,
        gap_type: str,
        numerator_label: str,
        numerator_value: float | None,
        denominator_label: str,
        denominator_value: float | None,
        notes: str,
    ) -> None:
        if numerator_value is None or denominator_value is None:
            return
        row = {
            "model_name_norm": model_name,
            "gap_type": gap_type,
            "numerator_label": numerator_label,
            "numerator_score_pct": round(numerator_value, 2),
            "denominator_label": denominator_label,
            "denominator_score_pct": round(denominator_value, 2),
            "gap_score_pct": round(numerator_value - denominator_value, 2),
            "gap_direction": "positive" if numerator_value - denominator_value >= 0 else "negative",
            "notes": notes,
        }
        attach_model_meta(row, model_name, model_meta)
        out.append(row)

    for model_name in models:
        moral_low = parse_float(headline_lookup.get((model_name, "MoralChoice", "Low Ambiguity Accuracy"), {}).get("ability_score_pct"))
        moral_high = parse_float(headline_lookup.get((model_name, "MoralChoice", "High Ambiguity Accuracy"), {}).get("ability_score_pct"))
        add_gap(
            model_name,
            "moralchoice_gap",
            "MoralChoice Low Ambiguity",
            moral_low,
            "MoralChoice High Ambiguity",
            moral_high,
            "Higher positive values mean the model is much better on clear-cut dilemmas than ambiguous ones.",
        )

        tom_unexpected = parse_float(headline_lookup.get((model_name, "Theory of Mind", "Unexpected Content"), {}).get("ability_score_pct"))
        tom_transfer = parse_float(headline_lookup.get((model_name, "Theory of Mind", "Transfer DQA"), {}).get("ability_score_pct"))
        add_gap(
            model_name,
            "tom_gap",
            "Theory of Mind Unexpected Content",
            tom_unexpected,
            "Theory of Mind Transfer DQA",
            tom_transfer,
            "Higher positive values mean the model handles basic false-belief style tasks better than transfer/generalization.",
        )

        emo_understanding = parse_float(headline_lookup.get((model_name, "EmoBench", "Emotion Understanding"), {}).get("ability_score_pct"))
        emo_application = parse_float(headline_lookup.get((model_name, "EmoBench", "Emotion Application"), {}).get("ability_score_pct"))
        add_gap(
            model_name,
            "emo_gap",
            "EmoBench Emotion Understanding",
            emo_understanding,
            "EmoBench Emotion Application",
            emo_application,
            "Higher positive values mean the model recognizes emotion more easily than it applies emotion in response.",
        )

        knowledge_score = parse_float(benchmark_lookup.get((model_name, "CBT-Bench"), {}).get("benchmark_score_pct"))
        diffdx_score = parse_float(benchmark_lookup.get((model_name, "MentalBench"), {}).get("benchmark_score_pct"))
        add_gap(
            model_name,
            "clinical_reasoning_gap",
            "CBT-Bench task knowledge",
            knowledge_score,
            "MentalBench differential diagnosis",
            diffdx_score,
            "Higher positive values mean the model is stronger on task knowledge/recall than on differential diagnosis.",
        )

        single_turn_scores: list[float] = []
        multi_turn_scores: list[float] = []
        for row in rows_starting_with_headline(summary_rows):
            if row["entity_label"] != model_name:
                continue
            pct = parse_float(row.get("ability_score_pct"))
            if pct is None:
                continue
            turn_type = str(row.get("single_turn_vs_multi_turn", "")).strip().lower()
            if turn_type == "single-turn":
                single_turn_scores.append(pct)
            elif turn_type == "multi-turn":
                multi_turn_scores.append(pct)
        add_gap(
            model_name,
            "single_vs_multi_turn_gap",
            "Single-turn mean",
            mean(single_turn_scores) if single_turn_scores else None,
            "Multi-turn mean",
            mean(multi_turn_scores) if multi_turn_scores else None,
            "Higher positive values mean the model performs better on single-turn than multi-turn psych evaluations.",
        )

    out.sort(key=lambda r: (r["gap_type"], -(parse_float(r.get("gap_score_pct")) or 0), r["model_name_norm"]))
    return out


def build_gap_summary(gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gap_rows:
        grouped[row["gap_type"]].append(row)

    expected_gap_types = [
        "moralchoice_gap",
        "tom_gap",
        "emo_gap",
        "clinical_reasoning_gap",
        "single_vs_multi_turn_gap",
    ]
    out: list[dict[str, Any]] = []
    for gap_type in expected_gap_types:
        rows = grouped.get(gap_type, [])
        values = [parse_float(r.get("gap_score_pct")) for r in rows]
        values = [v for v in values if v is not None]
        if not values:
            out.append({
                "gap_type": gap_type,
                "n_models": 0,
                "mean_gap_pct": "",
                "median_gap_pct": "",
                "min_gap_pct": "",
                "max_gap_pct": "",
                "sd_gap_pct": "",
                "aa_correlation_n": 0,
                "aa_pearson_r": "",
                "aa_pearson_p": "",
                "healthbench_best_available_n": 0,
                "healthbench_best_available_pearson_r": "",
                "healthbench_best_available_pearson_p": "",
            })
            continue
        stats = correlation_stats([
            (parse_float(r.get("gap_score_pct")) or 0.0, parse_float(r.get("aa_intelligence_index")) or 0.0)
            for r in rows
            if parse_float(r.get("gap_score_pct")) is not None and parse_float(r.get("aa_intelligence_index")) is not None
        ])
        hb_stats = correlation_stats([
            (parse_float(r.get("gap_score_pct")) or 0.0, parse_float(r.get("healthbench_best_available")) or 0.0)
            for r in rows
            if parse_float(r.get("gap_score_pct")) is not None and parse_float(r.get("healthbench_best_available")) is not None
        ])
        out.append({
            "gap_type": gap_type,
            "n_models": len(values),
            "mean_gap_pct": round(mean(values), 2),
            "median_gap_pct": round(median(values), 2),
            "min_gap_pct": round(min(values), 2),
            "max_gap_pct": round(max(values), 2),
            "sd_gap_pct": round(pstdev(values), 2) if len(values) > 1 else 0.0,
            "aa_correlation_n": stats["n"],
            "aa_pearson_r": stats["pearson_r"],
            "aa_pearson_p": stats["pearson_p"],
            "healthbench_best_available_n": hb_stats["n"],
            "healthbench_best_available_pearson_r": hb_stats["pearson_r"],
            "healthbench_best_available_pearson_p": hb_stats["pearson_p"],
        })
    return out


def coverage_tier(benchmark_count: int | None) -> str:
    count = benchmark_count or 0
    if count >= 7:
        return "complete_profile"
    if count >= 4:
        return "profile"
    if count >= 2:
        return "partial"
    if count >= 1:
        return "single_benchmark"
    return "no_coverage"


def main() -> None:
    benchmark_rows = load_csv("benchmark_results_long.csv")
    model_rows = load_csv("model_metadata.csv")
    frontier_rows = load_csv("frontier_healthbench_comparison.csv")
    supplemental_rows = load_csv("supplemental_findings.csv")
    model_meta = build_model_metadata(model_rows)

    benchmark_map_rows = build_benchmark_map()
    summary_rows = build_summary_rows(benchmark_rows, supplemental_rows, model_meta)
    normalize_group_percentiles(summary_rows)
    attach_raw_ability_scores(summary_rows)
    analysis_summary_rows = dedupe_summary_rows(summary_rows)
    normalize_group_percentiles(analysis_summary_rows)
    attach_raw_ability_scores(analysis_summary_rows)
    benchmark_composite_rows = build_benchmark_composite(analysis_summary_rows, model_meta)
    jagged_rows = build_jagged_profile(analysis_summary_rows, model_meta)
    model_psych_rows = build_model_psych_composite(benchmark_composite_rows, jagged_rows, model_meta)
    gap_rows = build_gap_rows(analysis_summary_rows, benchmark_composite_rows, jagged_rows, model_meta)
    gap_summary_rows = build_gap_summary(gap_rows)
    model_signal_rows = build_model_signal_composite(model_psych_rows, jagged_rows, gap_rows)
    correlation_rows = build_correlation_rows(model_psych_rows, jagged_rows, model_signal_rows)
    frontier_reference_rows = build_frontier_reference(frontier_rows, model_meta)
    frontier_timeline_rows = build_frontier_timeline(frontier_reference_rows, model_signal_rows)
    frontier_monthly_summary_rows = build_frontier_monthly_summary(frontier_timeline_rows)

    benchmark_field_order = [
        "benchmark",
        "benchmark_variant",
        "domain",
        "task_type",
        "metric_type",
        "higher_is_better",
        "single_turn_vs_multi_turn",
        "ambiguity_level",
        "clinical_risk_level",
        "family_tag",
        "summary_role",
        "notes",
    ]
    summary_field_order = [
        "benchmark",
        "benchmark_variant",
        "summary_role",
        "domain",
        "task_type",
        "metric_type",
        "higher_is_better",
        "single_turn_vs_multi_turn",
        "ambiguity_level",
        "clinical_risk_level",
        "family_tag",
        "entity_type",
        "entity_label",
        "model_name_norm",
        "provider",
        "model_family",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "aa_model_name",
        "aa_intelligence_index",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "source_type",
        "source_label",
        "source_title",
        "source_url",
        "benchmark_group",
        "benchmark_source",
        "dimension",
        "metric",
        "rank",
        "prompt_variant",
        "temperature",
        "mae_per_item",
        "stat_sd",
        "stat_min_item",
        "stat_r",
        "stat_p",
        "stat_n",
        "score_value",
        "score_text",
        "ability_score_pct",
        "normalized_pct",
        "score_min",
        "score_max",
        "score_range",
        "notes",
    ]
    analysis_summary_field_order = summary_field_order + [
        "score_value_min_observed",
        "score_value_max_observed",
        "source_count",
        "duplicate_row_count",
    ]
    benchmark_composite_field_order = [
        "model_name_norm",
        "provider",
        "model_family",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "aa_intelligence_index",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "healthbench_best_available",
        "benchmark",
        "domain",
        "task_type",
        "single_turn_vs_multi_turn",
        "ambiguity_level",
        "clinical_risk_level",
        "benchmark_score_pct",
        "benchmark_score_pct_min",
        "benchmark_score_pct_max",
        "benchmark_score_raw_mean",
        "benchmark_variant_count",
        "benchmark_variant_list",
        "family_tag_list",
    ]
    jagged_field_order = [
        "capability_family",
        "model_name_norm",
        "provider",
        "model_family",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "aa_intelligence_index",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "healthbench_best_available",
        "family_score_pct",
        "family_score_min_pct",
        "family_score_max_pct",
        "benchmark_count",
        "benchmark_list",
    ]
    model_psych_field_order = [
        "model_name_norm",
        "provider",
        "model_family",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "aa_intelligence_index",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "healthbench_best_available",
        "overall_psych_score_pct",
        "overall_psych_score_pct_min",
        "overall_psych_score_pct_max",
        "psych_score_variability_sd",
        "score_definition",
        "benchmark_count",
        "family_count",
        "coverage_tier",
        "main_visual_eligible",
        "complete_profile_eligible",
        "capability_family_list",
        "benchmark_list",
    ]
    model_signal_field_order = [
        "model_name_norm",
        "overall_psych_score_pct",
        "overall_psych_percentile_pct",
        "family_score_range_pct",
        "gap_abs_mean_pct",
        "family_range_percentile_pct",
        "gap_abs_percentile_pct",
        "jaggedness_score_pct",
        "stability_score_pct",
        "psych_balance_score_pct",
        "benchmark_count",
        "family_count",
        "score_definition",
        "coverage_tier",
        "main_visual_eligible",
        "complete_profile_eligible",
        "jaggedness_eligible",
        "provider",
        "model_family",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "log_params",
        "aa_intelligence_index",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "healthbench_best_available",
    ]
    correlation_field_order = [
        "analysis_scope",
        "scope_label",
        "psych_metric",
        "external_metric",
        "external_metric_label",
        "n",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
    ]
    gap_field_order = [
        "model_name_norm",
        "provider",
        "model_family",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "aa_intelligence_index",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "healthbench_best_available",
        "gap_type",
        "numerator_label",
        "numerator_score_pct",
        "denominator_label",
        "denominator_score_pct",
        "gap_score_pct",
        "gap_direction",
        "notes",
    ]
    gap_summary_field_order = [
        "gap_type",
        "n_models",
        "mean_gap_pct",
        "median_gap_pct",
        "min_gap_pct",
        "max_gap_pct",
        "sd_gap_pct",
        "aa_correlation_n",
        "aa_pearson_r",
        "aa_pearson_p",
        "healthbench_best_available_n",
        "healthbench_best_available_pearson_r",
        "healthbench_best_available_pearson_p",
    ]
    frontier_field_order = [
        "model_name_norm",
        "family",
        "provider",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "benchmark_date",
        "benchmark_date_note",
        "thinking_mode",
        "thinking_mode_note",
        "open_closed",
        "aa_intelligence_index",
        "aa_source",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "healthbench_source",
        "notes",
    ]
    frontier_timeline_field_order = [
        "model_name_norm",
        "family",
        "provider",
        "model_type",
        "parameter_size_billions",
        "release_date",
        "release_year",
        "release_month",
        "release_month_index",
        "release_month_key",
        "thinking_mode",
        "open_closed",
        "aa_intelligence_index",
        "healthbench_overall",
        "healthbench_consensus_overall",
        "benchmark_count",
        "overall_psych_score_pct",
        "jaggedness_score_pct",
        "stability_score_pct",
        "psych_balance_score_pct",
    ]
    frontier_monthly_summary_field_order = [
        "release_month_key",
        "release_year",
        "release_month",
        "release_month_index",
        "frontier_model_count",
        "model_list",
        "mean_parameter_size_billions",
        "mean_aa_intelligence_index",
        "mean_overall_psych_score_pct",
        "mean_jaggedness_score_pct",
        "mean_stability_score_pct",
        "mean_psych_balance_score_pct",
    ]

    write_csv_json("benchmark_map", benchmark_map_rows, benchmark_field_order)
    write_csv_json("model_benchmark_summary", summary_rows, summary_field_order)
    write_csv_json("analysis_summary_deduped", analysis_summary_rows, analysis_summary_field_order)
    write_csv_json("benchmark_composite", benchmark_composite_rows, benchmark_composite_field_order)
    write_csv_json("jagged_profile", jagged_rows, jagged_field_order)
    write_csv_json("model_psych_composite", model_psych_rows, model_psych_field_order)
    write_csv_json("model_signal_composite", model_signal_rows, model_signal_field_order)
    write_csv_json("correlation_results", correlation_rows, correlation_field_order)
    write_csv_json("gap_metrics", gap_rows, gap_field_order)
    write_csv_json("gap_summary", gap_summary_rows, gap_summary_field_order)
    write_csv_json("frontier_reference", frontier_reference_rows, frontier_field_order)
    write_csv_json("frontier_timeline", frontier_timeline_rows, frontier_timeline_field_order)
    write_csv_json("frontier_monthly_summary", frontier_monthly_summary_rows, frontier_monthly_summary_field_order)

    summary = {
        "benchmark_map_rows": len(benchmark_map_rows),
        "model_benchmark_summary_rows": len(summary_rows),
        "analysis_summary_deduped_rows": len(analysis_summary_rows),
        "benchmark_composite_rows": len(benchmark_composite_rows),
        "jagged_profile_rows": len(jagged_rows),
        "model_psych_composite_rows": len(model_psych_rows),
        "model_signal_composite_rows": len(model_signal_rows),
        "correlation_results_rows": len(correlation_rows),
        "gap_metrics_rows": len(gap_rows),
        "gap_summary_rows": len(gap_summary_rows),
        "frontier_reference_rows": len(frontier_reference_rows),
        "frontier_timeline_rows": len(frontier_timeline_rows),
        "frontier_monthly_summary_rows": len(frontier_monthly_summary_rows),
    }
    (DATA_DIR / "analysis_tables_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
