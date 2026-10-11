"""
eval_emergency.py — AuraVoice Emergency Detection Evaluation Harness
===================================================================
Tests emergency classifier directly against real independent utterances.
No Streamlit dependency, pure Python.
Reports:
- Tier-1 and Tier-2 recall with Wilson Score 95% Confidence Intervals
- Any-alert precision and per-tier precision
- 3x3 confusion matrix (Expected 0/1/2 vs Predicted 0/1/2)
- "Wrong-name / high-confidence" error cases
"""

import csv
import math
import sys
from collections import defaultdict

# Import pure emergency module
from emergency import (
    classify,
    Decision,
    resolve_rule_tier,
    evaluate_rule_prediction,
)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion p over n trials."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def run_evaluation(csv_path: str = "eval_emergency.csv"):
    results = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            lang = row["language"].strip()
            expected = int(row["expected_tier"].strip())

            # 1. Pure classifier decision
            dec = classify(text)

            # Use canonical post-classification resolver from emergency.py (prevents logic drift)
            rule_pred = evaluate_rule_prediction(text)

            results.append({
                "text": text,
                "language": lang,
                "expected": expected,
                "predicted_rule": rule_pred,
                "matched": dec.matched,
                "crisis": dec.crisis,
                "snakebite": dec.snakebite,
                "info_query": dec.info_query,
            })

    print("=" * 65)
    print("      AURAVOICE EMERGENCY EVALUATION REPORT (RULE ENGINE)")
    print("=" * 65)
    print(f"Total Test Utterances: {len(results)}\n")

    # 3x3 Confusion Matrix: [expected][predicted]
    conf_matrix = [[0, 0, 0] for _ in range(3)]
    for r in results:
        exp = r["expected"]
        prd = r["predicted_rule"]
        prd_clamped = min(max(prd, 0), 2)
        conf_matrix[exp][prd_clamped] += 1

    print("Confusion Matrix (Rows = Expected [0,1,2], Cols = Predicted [0,1,2]):")
    print("             Pred 0   Pred 1   Pred 2")
    for row_idx, row_vals in enumerate(conf_matrix):
        print(f"Expected {row_idx}:   {row_vals[0]:6d}   {row_vals[1]:6d}   {row_vals[2]:6d}")
    print()

    # Per-language and overall metrics
    for lang in ["all", "en", "hi"]:
        subset = [r for r in results if lang == "all" or r["language"] == lang]
        if not subset:
            continue

        tp1 = sum(1 for r in subset if r["expected"] == 1 and r["predicted_rule"] == 1)
        fn1 = sum(1 for r in subset if r["expected"] == 1 and r["predicted_rule"] != 1)
        n_t1 = tp1 + fn1

        tp2 = sum(1 for r in subset if r["expected"] == 2 and r["predicted_rule"] == 2)
        fn2 = sum(1 for r in subset if r["expected"] == 2 and r["predicted_rule"] != 2)
        n_t2 = tp2 + fn2

        # Any alert True Positives & False Positives
        tp_any = sum(1 for r in subset if r["expected"] > 0 and r["predicted_rule"] > 0)
        fp_any = sum(1 for r in subset if r["expected"] == 0 and r["predicted_rule"] > 0)
        tn_any = sum(1 for r in subset if r["expected"] == 0 and r["predicted_rule"] == 0)
        fn_any = sum(1 for r in subset if r["expected"] > 0 and r["predicted_rule"] == 0)

        rec_t1 = tp1 / n_t1 if n_t1 > 0 else 0.0
        ci_t1 = wilson_ci(rec_t1, n_t1)

        rec_t2 = tp2 / n_t2 if n_t2 > 0 else 0.0
        ci_t2 = wilson_ci(rec_t2, n_t2)

        any_precision = tp_any / (tp_any + fp_any) if (tp_any + fp_any) > 0 else 0.0
        ci_prec = wilson_ci(any_precision, tp_any + fp_any)

        print(f"--- Language: {lang.upper()} (N={len(subset)}) ---")
        print(f"  Tier-1 Recall (CALL NOW) : {rec_t1:6.1%}  95% CI [{ci_t1[0]:.1%} - {ci_t1[1]:.1%}] (TP={tp1}, FN={fn1})")
        print(f"  Tier-2 Recall (URGENT)   : {rec_t2:6.1%}  95% CI [{ci_t2[0]:.1%} - {ci_t2[1]:.1%}] (TP={tp2}, FN={fn2})")
        print(f"  Any-Alert Precision      : {any_precision:6.1%}  95% CI [{ci_prec[0]:.1%} - {ci_prec[1]:.1%}] (TP={tp_any}, FP={fp_any})")
        print(f"  True Negatives           : {tn_any} / {tn_any + fp_any} ({tn_any/(tn_any+fp_any):.1%})\n")

    # Wrong-name / High-confidence error analysis
    wrong_name = [
        r for r in results
        if (r["expected"] == 0 and r["predicted_rule"] > 0)
        or (r["expected"] > 0 and r["predicted_rule"] > 0 and r["expected"] != r["predicted_rule"])
    ]
    print(f"Wrong-name / Cross-tier discrepancies: {len(wrong_name)}")
    for wn in wrong_name:
        print(f"  - \"{wn['text']}\" -> Expected {wn['expected']}, Got {wn['predicted_rule']} (Matched: '{wn['matched']}')")

    return results


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "eval_emergency.csv"
    run_evaluation(csv_file)
