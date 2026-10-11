"""
eval_llm_veto.py — AuraVoice LLM Emergency Veto & Safety Layer Evaluation
========================================================================
Measures the full Rule + LLM confirmation/veto pipeline by calling
the actual llm_emergency_confirm and llm_emergency_check functions.

Key evaluation metrics:
1. Precision and recall before vs. after LLM veto layer.
2. Veto accuracy: How many false alarms were correctly suppressed.
3. Veto safety: Whether any TRUE Tier-2 emergencies were erroneously removed (adverse suppression).
4. Second-layer upgrade: Whether LLM caught rule-engine false negatives.

Usage:
  python eval_llm_veto.py [csv_path] [--output results.csv]
"""

import csv
import math
import os
import sys
from typing import Optional

from emergency import (
    classify,
    Decision,
    resolve_rule_tier,
    evaluate_rule_prediction,
)
from app import (
    init_gemini,
    init_groq,
    llm_emergency_confirm,
    llm_emergency_check,
    _get_secrets,
)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion p over n trials."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def run_veto_evaluation(csv_path: str = "eval_emergency.csv", output_csv: Optional[str] = None):
    # Initialize real LLM clients from secrets / environment
    gk_secret, grk_secret = _get_secrets()
    gemini_key = os.getenv("GEMINI_API_KEY", gk_secret).strip()
    groq_key = os.getenv("GROQ_API_KEY", grk_secret).strip()

    gemini_client = init_gemini(gemini_key) if gemini_key else None
    groq_client = init_groq(groq_key) if groq_key else None

    has_live_llm = (gemini_client is not None or groq_client is not None)
    if not has_live_llm:
        print("[WARNING] No GEMINI_API_KEY or GROQ_API_KEY found in environment or secrets.toml.")
        print("[WARNING] Running with mock fallback (simulated model responses).")

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "text": r["text"].strip(),
                "language": r.get("language", "en").strip(),
                "expected": int(r["expected_tier"].strip()),
            })

    results = []
    t2_candidates = 0
    t2_true_cases_vetoed = 0
    t2_false_alarms_vetoed = 0

    print("=" * 68)
    print(f"  AURAVOICE LLM VETO & SAFETY LAYER EVALUATION ({csv_path})")
    print(f"  Live Model Calling: {'ENABLED' if has_live_llm else 'SIMULATED (No keys)'}")
    print("=" * 68)

    for item in rows:
        text = item["text"]
        lang = item["language"]
        expected = item["expected"]

        dec = classify(text)
        rule_pred = evaluate_rule_prediction(text)

        final_pred = rule_pred
        veto_action = "none"

        # Check if candidate for LLM confirmation / veto (Tier 2 non-crisis non-snakebite)
        if rule_pred == 2 and not dec.crisis and not dec.snakebite:
            t2_candidates += 1
            if has_live_llm:
                confirmed = llm_emergency_confirm(text, gemini_client, groq_client)
            else:
                # Simulated heuristic for offline environments without API keys:
                # Negations like 'nahi' or 'do not' are vetoed
                lower = text.lower()
                confirmed = not ("nahi" in lower or "not " in lower or "no " in lower)

            if not confirmed:
                final_pred = 0
                veto_action = "vetoed_to_0"
                if expected == 2:
                    t2_true_cases_vetoed += 1
                elif expected == 0:
                    t2_false_alarms_vetoed += 1
            else:
                veto_action = "confirmed"

        # Check if candidate for Second-Layer Emergency Catch (Tier 0 -> 2)
        elif rule_pred == 0 and not dec.info_query:
            if has_live_llm:
                upgrade_chk = llm_emergency_check(text, gemini_client, groq_client)
                if upgrade_chk is True:
                    final_pred = 2
                    veto_action = "upgraded_to_2"

        results.append({
            "text": text,
            "language": lang,
            "expected": expected,
            "rule_pred": rule_pred,
            "final_pred": final_pred,
            "matched_phrase": dec.matched,
            "crisis": dec.crisis,
            "snakebite": dec.snakebite,
            "veto_action": veto_action,
        })

    # Summary metrics
    n_total = len(results)
    tp_rule = sum(1 for r in results if r["expected"] > 0 and r["rule_pred"] > 0)
    fp_rule = sum(1 for r in results if r["expected"] == 0 and r["rule_pred"] > 0)
    prec_rule = tp_rule / (tp_rule + fp_rule) if (tp_rule + fp_rule) > 0 else 0.0

    tp_final = sum(1 for r in results if r["expected"] > 0 and r["final_pred"] > 0)
    fp_final = sum(1 for r in results if r["expected"] == 0 and r["final_pred"] > 0)
    prec_final = tp_final / (tp_final + fp_final) if (tp_final + fp_final) > 0 else 0.0

    print(f"\nTotal Utterances Evaluated: {n_total}")
    print(f"\n--- Veto Performance on Tier-2 Candidates ---")
    print(f"  Tier-2 candidates inspected by LLM    : {t2_candidates}")
    print(f"  False alarms successfully vetoed (FP->0): {t2_false_alarms_vetoed}")
    print(f"  True Tier-2 emergencies removed (TP->0) : {t2_true_cases_vetoed}  <-- Adverse suppression check")

    print(f"\n--- Comparative Performance ---")
    print(f"  Rule-Only Precision : {prec_rule:6.1%} ({tp_rule}/{tp_rule + fp_rule})")
    print(f"  Rule + Veto Precision: {prec_final:6.1%} ({tp_final}/{tp_final + fp_final})")

    # Optional save to CSV
    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "text", "language", "expected", "rule_pred", "final_pred",
                "matched_phrase", "crisis", "snakebite", "veto_action"
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed log written to: {output_csv}")

    return results


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "eval_emergency.csv"
    out_file = "eval_veto_output.csv"
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            out_file = sys.argv[idx + 1]
    run_veto_evaluation(target_csv, out_file)
