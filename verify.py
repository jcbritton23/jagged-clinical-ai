"""Verify review-table coverage and replay the article's observed-model comparisons."""
import csv
from collections import Counter, defaultdict
from math import isclose
from pathlib import Path
from statistics import correlation, mean

ROOT = Path(__file__).resolve().parent


def read_table(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


AUDIT_RESULTS = ("match", "qualified", "differs", "invalid", "unresolved")


def verify_audit(benchmarks, supplemental):
    tables = [
        (
            benchmarks,
            "review/primary-source-checks.csv",
            {
                "benchmark": "benchmark",
                "model_as_recorded": "model_name_raw",
                "size_as_recorded": "size_raw",
                "dimension": "dimension",
                "archived_value": "score_value",
                "archived_text": "score_text",
            },
        ),
        (
            supplemental,
            "review/supplemental-source-checks.csv",
            {
                "benchmark_or_framework": "benchmark_or_framework",
                "model_or_subject": "model_or_subject",
                "metric": "metric",
                "archived_value": "value",
            },
        ),
    ]
    for source, path, fields in tables:
        ledger = read_table(path)
        assert len(ledger) == len(source)
        for csv_row, (record, check) in enumerate(zip(source, ledger), start=2):
            assert int(check["csv_row"]) == csv_row
            assert all(check[output] == record[input_] for output, input_ in fields.items())
            assert check["result"] in AUDIT_RESULTS
            assert check["primary_source"] and check["locator"]
        counts = Counter(check["result"] for check in ledger)
        print(f"{path}: " + ", ".join(f"{counts[result]} {result}" for result in AUDIT_RESULTS))

    checks = Counter((check["benchmark"], check["result"])
                     for check in read_table("review/primary-source-checks.csv"))
    coverage = read_table("review/coverage.csv")
    assert Counter(row["benchmark"] for row in benchmarks) == {
        row["benchmark"]: int(row["archive_rows"]) for row in coverage
    }
    assert all(checks[row["benchmark"], result] == int(row[result])
               for row in coverage for result in AUDIT_RESULTS)


def verify():
    master = read_table("data/master_observations.csv")
    benchmarks = read_table("data/benchmark_results_long.csv")
    assert Counter(row["record_type"] for row in master) == {"benchmark": 1320, "supplemental": 150}
    assert len({tuple(row.items()) for row in master}) == len(master)
    columns = ["benchmark", "model_name_raw", "dimension", "score_text", "source_label", "provenance_detail"]
    def records(rows):
        return Counter(tuple(row[k] for k in columns) for row in rows)
    assert records(benchmarks) == records(row for row in master if row["record_type"] == "benchmark")
    supplemental = read_table("data/supplemental_findings.csv")
    master_supplemental = [row for row in master if row["record_type"] == "supplemental"]
    supplement_columns = ("benchmark_or_framework", "model_or_subject", "metric", "value", "unit", "source_url")
    assert Counter(tuple(row[k] for k in supplement_columns) for row in supplemental) == Counter(
        tuple(row[k] for k in supplement_columns) for row in master_supplemental
    )
    verify_audit(benchmarks, supplemental)
    rows = read_table("data/blog_benchmark_rows_for_visuals.csv")
    scores = defaultdict(list)
    labels = defaultdict(set)
    for row in rows:
        key = row["model_name_norm"], row["capability_family"]
        score = float(row["visual_score_pct"])
        assert 0 <= score <= 100
        scores[key].append(score)
        labels[key].add(row["benchmark_label"])
    profile = read_table("data/blog_jagged_profile_moral_combined.csv")
    assert len(scores) == len(profile) == 254
    by_model = defaultdict(list)
    counts = Counter()
    for row in profile:
        model = row["model_name_norm"]
        key = model, row["capability_family"]
        value = mean(scores[key])
        assert isclose(value, float(row["family_score_pct"]), abs_tol=1e-9)
        assert len(labels[key]) == int(row["benchmark_count"])
        by_model[model].append(value)
        counts[model] += len(labels[key])
    summary = read_table("data/blog_score_review_moral_combined_4plus.csv")
    assert {r["model_name_norm"] for r in summary} == {m for m, n in counts.items() if n >= 4}
    for row in summary:
        model = row["model_name_norm"]
        values = by_model[model]
        assert len(values) == int(row["family_count"]) and counts[model] == int(row["benchmark_count"])
        assert isclose(mean(values), float(row["overall_psych_score_pct"]), abs_tol=1e-9)
        assert isclose(max(values)-min(values), float(row["jaggedness_score_pct"]), abs_tol=1e-9)
    for minimum, expected in [(1,(13,0.484268,-0.059686)), (4,(9,0.614085,-0.222737))]:
        matched = [r for r in summary if r["aa_intelligence_index"] and int(r["family_count"]) >= minimum]
        aa = [float(r["aa_intelligence_index"]) for r in matched]
        result = [correlation(aa, [float(r[k]) for r in matched])
                  for k in ["overall_psych_score_pct","jaggedness_score_pct"]]
        assert len(matched) == expected[0]
        assert all(isclose(a,b,abs_tol=1e-6) for a,b in zip(result, expected[1:]))
        print(f"{len(matched)} matched models: score r={result[0]:.4f}, range r={result[1]:.4f}")
    print(f"Verified {len(master)} master observations and {len(profile)} family profiles.")


if __name__ == "__main__":
    verify()
