# EVALUATION.md — AuraVoice Evaluation Protocol

Run these tests before making credibility claims in your project documentation.
Results should be reported at the top of your README with the date and evaluator.

---

## 1. Emergency Detection — Precision & Recall

### Goal
Measure how often the rule-based + LLM-confirm pipeline correctly identifies
genuine emergencies (recall) and does not fire on non-emergencies (precision).

### Test Set Construction (aim for 100–150 utterances)

Collect test cases from **real interactions, not from the seed phrase lists** in `app.py`.
Using the seed list to test itself inflates recall artificially.

**Positive set (~60 utterances) — should trigger alert:**
- 20 clear English emergencies (cardiac, respiratory, stroke, snakebite, mental health)
  — use paraphrases, not verbatim seed phrases
- 20 clear Hindi emergencies (mix of Devanagari and transliteration)
  — include real patient-voice phrasings, not just seed echoes
- 10 regional language examples (Bhojpuri, Awadhi, Bengali, Tamil, Telugu)
  — **do not count these toward EN/HI recall figures; report separately as "seed coverage"**
- 10 boundary cases (uncommon phrasing, code-switching, typos)

**Negative set (~60 utterances) — should NOT trigger tier-1:**
- 15 genuine non-emergencies using emergency words
  - "mujhe kabhi kabhi seene mein dard nahi hota" (I never get chest pain)
  - "help me understand this label"
  - "my heart is full of gratitude"
  - "fitting room is on the left"
  - "I had a stroke of luck"
  - "seene mein dard nahi hai aaj" — ← **expected tier-0** (no current pain)
- 15 routine symptom descriptions (headache, cold, mild fever)
- 15 medication questions
- 15 context-free greetings / app navigation questions

> **Important:** Also create a "wrong-name / high-confidence" category:
> cases where the model fires with high confidence but matches the wrong phrase.
> Report this error class separately in your results.

### Tier-1 vs Tier-2 Recall
Report recall **separately** for tier-1 (CALL NOW) and tier-2 (URGENT):
- Tier-1 false negatives (missed genuine CALL NOW) are the highest-risk error.
- Tier-2 false positives are alert fatigue but not safety-critical.

### Running the Test

Save your utterances in a CSV with columns: `text,language,expected_tier`. Run with:
```bash
python eval_emergency.py
```

Or execute the harness script directly:

```python
import csv, math, sys
from emergency import classify, is_strong_past_tense, Decision

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion p over n trials."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

# Load test dataset
results = []
with open("eval_emergency.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text     = row["text"].strip()
        lang     = row["language"].strip()
        expected = int(row["expected_tier"].strip())

        # Pure emergency classifier
        dec = classify(text)

        # Fold crisis into predicted tier >= 2 (crisis never downgraded)
        # Apply strong past-tense check (tier > 0 downgraded to 0 for strong historical events)
        if dec.tier > 0 and not dec.crisis and is_strong_past_tense(text):
            predicted = 0
        elif dec.info_query:
            predicted = 0  # informational query: answered with disclaimer banner
        elif dec.crisis:
            predicted = max(dec.tier, 2)
        else:
            predicted = dec.tier

        results.append({
            "text":      text,
            "language":  lang,
            "expected":  expected,
            "predicted": predicted,
            "matched":   dec.matched,
            "crisis":    dec.crisis,
            "snakebite": dec.snakebite,
            "info_query": dec.info_query,
        })

# 3x3 Tier Confusion Matrix: [expected][predicted]
conf_matrix = [[0, 0, 0] for _ in range(3)]
for r in results:
    exp = r["expected"]
    prd = min(max(r["predicted"], 0), 2)
    conf_matrix[exp][prd] += 1

print("\n=== 3x3 Confusion Matrix (Expected vs Predicted) ===")
print("             Pred 0   Pred 1   Pred 2")
for idx, row in enumerate(conf_matrix):
    print(f"Expected {idx}:   {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")

# Per-tier, per-language metrics (Rule-only and Rule+Veto reported separately)
for lang in ["all", "en", "hi"]:
    subset = [r for r in results if lang == "all" or r["language"] == lang]
    if not subset:
        continue
    tp1 = sum(1 for r in subset if r["expected"] == 1 and r["predicted"] == 1)
    fn1 = sum(1 for r in subset if r["expected"] == 1 and r["predicted"] != 1)
    tp2 = sum(1 for r in subset if r["expected"] == 2 and r["predicted"] == 2)
    fn2 = sum(1 for r in subset if r["expected"] == 2 and r["predicted"] != 2)
    
    tp_any = sum(1 for r in subset if r["expected"] > 0 and r["predicted"] > 0)
    fp_any = sum(1 for r in subset if r["expected"] == 0 and r["predicted"] > 0)
    tn_any = sum(1 for r in subset if r["expected"] == 0 and r["predicted"] == 0)

    recall_t1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
    recall_t2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    precision_any = tp_any / (tp_any + fp_any) if (tp_any + fp_any) > 0 else 0

    ci_r1 = wilson_ci(recall_t1, tp1 + fn1)
    ci_r2 = wilson_ci(recall_t2, tp2 + fn2)
    ci_p  = wilson_ci(precision_any, tp_any + fp_any)

    print(f"\n=== Language: {lang.upper()} (N={len(subset)}) ===")
    print(f"  Tier-1 recall:       {recall_t1:.1%}  95% CI [{ci_r1[0]:.1%}–{ci_r1[1]:.1%}]  (N={tp1+fn1})")
    print(f"  Tier-2 recall:       {recall_t2:.1%}  95% CI [{ci_r2[0]:.1%}–{ci_r2[1]:.1%}]  (N={tp2+fn2})")
    print(f"  Any-Alert Precision: {precision_any:.1%}  95% CI [{ci_p[0]:.1%}–{ci_p[1]:.1%}]")
    print(f"  True Negatives:      {tn_any} / {tn_any + fp_any} ({tn_any/(tn_any+fp_any):.1%})")

# Wrong-name / High-confidence errors
wrong_name = [r for r in results if r["expected"] == 0 and r["predicted"] > 0]
print(f"\nWrong-Name / Spurious Matches (Expected 0, Got >0): {len(wrong_name)}")
for wn in wrong_name:
    print(f"  • \"{wn['text']}\" -> Predicted Tier {wn['predicted']} (Matched: '{wn['matched']}')")
```

### Reporting
Add to README:
```
Emergency detection:
  Tier-1 recall (EN):  XX%  95% CI [XX%–XX%]  N=XX
  Tier-1 recall (HI):  XX%  95% CI [XX%–XX%]  N=XX
  Tier-2 recall (EN):  XX%
  Precision (EN):      XX%
  Tested on N utterances — Date  Evaluator
```

**Minimum bar before claiming multilingual emergency support:**
- English tier-1 recall ≥ 95% (a missed cardiac arrest is the worst failure mode)
- English tier-2 recall ≥ 85%, precision ≥ 80%
- Hindi tier-1 recall ≥ 90%
- Other languages: report as "seed coverage — not evaluated"

---

## 2. Prescription Reading — Drug Name Accuracy

### Goal
Measure how accurately AuraVoice reads drug names from real medicine labels.

### Test Set Construction (aim for 30 photos)

Photograph 30 medicine strips or boxes you have at home or can access at a pharmacy.
Vary:
- Lighting: bright, dim, fluorescent, phone torch
- Angle: straight-on, slight tilt, slight glare
- Label types: strip (small font), box (larger font), photocopied prescription
- Languages: English brand name, Hindi transliteration on label, bilingual

For each photo, manually note:
- Drug name (ground truth)
- Dose/strength (ground truth)
- Expiry date (ground truth)

### Running the Test

Process each image through AuraVoice (Vision Analysis → Prescription tab).
Record the model's output in a spreadsheet:

| Photo | GT Drug Name | Model Drug Name | Name Match | GT Dose | Model Dose | Dose Match | Confidence Reported | Quality Flag |
|---|---|---|---|---|---|---|---|---|
| img_001 | Metformin 500mg | Metformin 500mg | ✅ | 500mg | 500mg | ✅ | HIGH | None |
| img_002 | Amlodipine 5mg | Amtodipine 5mg | ❌ (OCR error) | 5mg | 5mg | ✅ | MEDIUM | blurry |

### Metrics to Report

```
Drug name accuracy:    XX / 30 (XX%)
Dose accuracy:         XX / 30 (XX%)
Expiry date accuracy:  XX / 30 (XX%)
Confidence calibration:
  HIGH labels:   XX% were actually correct
  MEDIUM labels: XX% were actually correct
  LOW labels:    XX% were actually correct
```

### Notes
- If the model says "Possible read: X — verify with pharmacist", count this as a
  correct detection of an OCR risk, even if the drug name extraction is wrong.
- Report separately for good-quality vs low-quality photos.
- Minimum bar: ≥80% name accuracy on good-quality photos.
- **No drug database validation is performed in this MVP** — the model cannot
  confirm the drug exists. This is a known limitation.

---

## 3. Latency Measurement

Time 10 queries of each type and report P50 and P90 in seconds.

**P90 calculation fix:** `sorted(times)[int(0.9 * n)]` with n=10 returns index 9 which is
the maximum. Use `sorted(times)[int(0.9 * (n - 1))]` or `statistics.quantiles`:

```python
import time, statistics

def measure_latency(fn, n=10):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        time.sleep(1)
    sorted_t = sorted(times)
    # statistics.quantiles is cleaner for small n
    p90_idx = min(int(0.9 * n), n - 1)    # clamp to valid index
    return {
        "p50": round(statistics.median(times), 2),
        "p90": round(sorted_t[p90_idx], 2),
        "min": round(min(times), 2),
        "max": round(max(times), 2),
    }
```

Report in README:
```
Latency (free API tier, India, Oct 2026):
  Symptom check (text, Gemini): P50 Xs  P90 Xs
  Prescription read (image):    P50 Xs  P90 Xs
  Emergency detection (rule):   P50 <0.01s  (no API call)
```

---

## 4. User Test — Minimal (5 sessions)

Even 5 family members counts. **Include at least one blind or low-vision participant.**
Choose:
- 1 person aged 60+
- 1 person who speaks primarily Hindi
- 1 person unfamiliar with smartphones
- 1 person with low vision or blindness (test TTS + accessibility path)
- 1 person in the 25–45 age range (baseline)

For each session, ask them to:
1. Read a prescription label (photograph their own medicine)
2. Describe a symptom (any recent ailment)
3. Find the emergency number for India
4. Use the audio tab to ask a health question

Record:
- Task completion: yes / no / with help
- Response comprehensibility: 1 (unclear) to 5 (very clear)
- TTS comprehensibility (for Hindi users): 1–5
- Any confusion with the UI

Report findings honestly, including failures.

---

## 5. What to Put at the Top of Your README

Once you have numbers, add this block just below the project description:

```markdown
## 📊 Evaluation Results (Month Year)

| Metric | Result | Notes |
|---|---|---|
| Emergency detection tier-1 recall (EN) | XX% [CI] | N=XX, from independent test set |
| Emergency detection tier-1 recall (HI) | XX% [CI] | N=XX |
| Emergency detection precision (EN)     | XX%       | |
| Drug name accuracy (good photos)       | XX%       | N=30 labels |
| Latency — symptom check P50            | Xs        | Gemini 3.6 Flash, free tier |
| User test task completion              | XX/5      | include participant demographics |
| Evaluator | [Your name] | [Date] |
```

---

## Seed CSV Template

Save as `eval_emergency.csv`. **Do not use these exact phrases as your full test set** —
they are the same phrases in the seed lists, so testing on them measures memorisation,
not generalisation. Use them only for sanity checks; collect novel paraphrases for
your actual evaluation.

```csv
text,language,expected_tier
"I think I am having a heart attack",en,1
"seene mein bahut tez dard ho raha hai",hi,1
"help me understand this medicine label",en,0
"seene mein dard nahi hai aaj",hi,0
"I have chest pain when climbing stairs",en,2
"saans lene mein takleef ho rahi hai",hi,2
"fitting room is on the right",en,0
"marna chahti hoon",hi,2
"my heart is full of happiness today",en,0
"saanp ne kata abhi",hi,1
"My father had a heart attack last year",en,0
"mere haath par koi cheez ne kata shayad saanp tha",hi,1
```

Add 100+ rows following this pattern. Partner with a Hindi speaker to create
authentic negative cases for Hindi. Report test set composition (source, annotator)
in your project documentation.
