# 🎙️ AuraVoice: Multimodal Health & Accessibility Agent

> *"What if your phone could be your doctor, your guide, and your safety net — all at once?"*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://auravoice-multimodal-health-accessibility-agent-7sc3pkcpirqquh.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🚀 **Live Deployment:** [https://auravoice-multimodal-health-accessibility-agent-7sc3pkcpirqquh.streamlit.app/](https://auravoice-multimodal-health-accessibility-agent-7sc3pkcpirqquh.streamlit.app/)  

---

## 📖 Project Description

**AuraVoice is a health information tool** — not a medical service — that turns a smartphone camera or microphone into a multilingual health literacy and accessibility assistant. A grandmother in rural Bihar can photograph her pill bottle and hear a plain-language explanation in Hindi. A patient in a remote area can describe symptoms by voice and receive structured triage guidance.

> **Honest scope:** AuraVoice is a triage and health literacy aid. It has not been clinically evaluated, validated by certified clinicians, or tested with patients at scale. All guidance should be verified by a qualified healthcare professional.

> **Note on statistics:** 1.3B people with disability ([WHO 2023](https://www.who.int/news-room/fact-sheets/detail/disability-and-health)), ~39M blind + 246M moderate/severe visual impairment ([WHO GBD 2020](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)), ~1.1B people aged 60+ globally ([WHO 2023](https://www.who.int/news-room/fact-sheets/detail/ageing-and-health)), ~900M rural population in India ([World Bank / Census](https://data.worldbank.org/indicator/SP.RUR.TOTL?locations=IN)) facing specialist shortages ([MoHFW Rural Health Statistics](https://main.mohfw.gov.in)). These are cited demographic figures, not claims attributable to this tool.

---

## 📊 Evaluation Results (October 2026)

### Rigorous Evaluation: In-Sample Development vs. Out-of-Sample Held-Out Benchmarks
To maintain clinical and technical integrity, **in-sample figures are never headlined alone**. Keyword matching exhibits a steep drop on unseen vocabulary, which is why AuraVoice runs an **LLM safety layer on every input** in the production app.

| Benchmark Set | Evaluation Type | N | Tier-1 Recall | Tier-2 Recall | Precision | True Negatives | Clinical Takeaway |
|---|---|---|---|---|---|---|---|
| **Held-Out Test Set 1 (`eval_heldout.csv`)** | Out-of-Sample (Blind) | 40 | **15.4%** (2/13) | **0.0%** (0/9) | **100.0%** (4/4) | **100.0%** (18/18) | Pure keyword generalization boundary on unseed phrases |
| **Held-Out Test Set 2 (`eval_heldout_set3.csv`)** | Out-of-Sample (Fresh) | 49 | **33.3%** (6/18) | **28.6%** (4/14) | **100.0%** (10/10) | **100.0%** (17/17) | Broadened rules capture key synonyms with 0 false alarms |
| **Development Benchmark (`eval_emergency.csv`)** | In-Sample (Tuned) | 75 | **100.0%** (27/27) | **83.3%** (10/12) | **86.0%** (37/43) | **83.3%** (30/36) | In-sample performance during rule authoring |

---

### Real Model Veto & Dual-Layer Safety Evaluation (`eval_llm_veto.py`)
To test the live application pipeline where the rule engine is combined with real model calls (`llm_emergency_confirm` and `llm_emergency_check` using Gemini 3.6 Flash / Groq):

- **False Alarm Suppression:** The LLM veto inspected 8 candidate Tier-2 alerts, successfully vetoing 3 conversational negations (*"seene mein dard nahi hai aaj"*), boosting any-alert precision from **90.2% (37/41)** to **97.4% (37/38)**.
- **Transparent Adverse Suppression Disclosure:** Precision alone hides sensitivity loss. The LLM veto erroneously removed **1 true Tier-2 emergency** (suppressed to non-alert), demonstrating the exact real-world trade-off of alert-fatigue filtering.
- **Second-Layer Safety Check:** On every input where the rule engine predicts Tier 0, `llm_emergency_check` evaluates whether an acute crisis was missed, ensuring coverage beyond keyword dictionaries.
- **Safety Invariant:** Tier-1 (CALL NOW), Crisis/self-harm, and Snakebite are **hard-locked** and can **never** be downgraded by the LLM.

### Offline & Live Benchmark Reproduction
```cmd
# 1. Deterministic offline rule evaluation (zero API keys required)
python eval_emergency.py eval_heldout.csv
python eval_emergency.py eval_heldout_set3.csv
python eval_emergency.py eval_emergency.csv

# 2. Live LLM Veto & Safety evaluation (requires API keys in secrets.toml or environment)
python eval_llm_veto.py eval_emergency.csv
```

> **Evaluation Harness vs. Live App Behavior:**
> In `eval_emergency.py`, the offline harness maps strong past-tense utterances (*"had a heart attack last week"*) and informational queries to non-alerts (predicted tier 0). In the interactive Streamlit app, both cases trigger safety UI: past-tense events prompt a confirmation check (*"Is this happening RIGHT NOW?"*), and informational queries render the emergency alert banner with a one-click dismissal button (*"Not an emergency"*).

---

## 🎯 Core Features

| Module | Description | Technology |
|---|---|---|
| **💊 Prescription Reader** | Image quality check → label reading with confidence (High/Medium/Low) → teach-back | Gemini 3.6 Flash Vision |
| **🏠 Environment Navigator** | Navigation guidance for blind users (clock positions, step counts, hazards) | Gemini 3.6 Flash Vision |
| **🔍 Symptom Analyser** | Structured triage: urgency level 🔴/🟡/🟢 + self-care + doctor questions | Gemini 3.6 Flash / Qwen3.8-27b |
| **🎙️ Voice Interface** | In-browser mic recording → STT → LLM response with TTS playback | streamlit-mic-recorder + Google STT |
| **🚨 Emergency Detection** | Tiered phrases across EN/HI/Bhojpuri/Awadhi/Bengali/Tamil/Telugu. No negation suppression. Crisis sub-tier with Tele-MANAS message. | Rule engine |
| **🏥 Doctor Handoff Summary** | SOAP-lite structured one-pager the patient can show their doctor | Gemini 3.6 Flash |
| **💊 Medication Reconciliation** | Discharge summary → daily schedule + duplicate/interaction/Beers flags + Jan Aushadhi generics | Gemini 3.6 Flash |
| **📷 Image Quality Gate** | Blur, glare, size checks before prescription reads — camera retake guidance | PIL edge-map variance |
| **🔊 Text-to-Speech** | Every response read aloud in user's language; sentence-chunked for reliability; Play/Stop in HTML | Web Speech API |
| **🔤 Simple Mode** | Large text, high contrast, bigger buttons — designed for elderly/low-vision users | CSS injection |
| **🌍 Multilingual** | Responds in user's language; Devanagari + transliteration in emergency set; missing-language banner | Gemini native |
| **🔁 3-Tier Fallback** | Gemini 3.6 Flash → Groq Qwen3.8-27b → static safe response | Resilient architecture |
| **📋 Consent Gate** | Informed consent gate shown before app access | Session-state gate |
| **🏥 India Health Pathways** | 112 / 108 / 102 / Tele-MANAS / iCall (hours-labelled) / Childline / Elderline / Poison Control | Static verified panel |

---

## 🌍 Real-World Impact

### Target Populations

| Population | Estimated Size | How AuraVoice Helps |
|---|---|---|
| Visually impaired | ~39M blind + 246M moderate/severe visual impairment (WHO 2020) | Environment navigation + prescription reading |
| Elderly with medication confusion | ~1.1B aged 60+ globally (WHO 2023) | Plain-language label explanation + medication schedule |
| Rural patients without doctor access | ~900M in India alone (Census / MoHFW) | Symptom triage + referral routing |
| Low health literacy patients | 1 billion+ | Jargon-free, structured guidance |
| Non-English speakers in healthcare | 2+ billion | Native-language responses |
| Post-discharge patients | Tens of millions annually | Medication reconciliation + daily schedule |

### What We Can Responsibly Claim

- The medication reconciliation module addresses the highest-risk moment in a care episode (hospital discharge), which is documented as a major source of adverse drug events (source: IHI, WHO Medication Safety Challenge).
- Emergency detection responds to matched phrases in under 1 second before any LLM call.
- The teach-back technique (asking users to repeat their schedule) is a validated health literacy intervention (AHRQ Health Literacy Universal Precautions Toolkit).
- Response latency on free API tiers is typically 2–6 seconds — not measured in real-time in this MVP.

> We do not claim to "replace a GP consultation" or cite medication error reduction rates as attributable to this tool. These would require clinical evaluation.

---

## 🚀 Innovation & Originality

### Why AuraVoice Is Different

| Dimension | Typical Health Apps | AuraVoice |
|---|---|---|
| **Input modality** | Text only | Camera + mic + text simultaneously |
| **Language coverage** | English-centric | EN/HI/Bhojpuri/Awadhi/Bengali/Tamil/Telugu emergency detection |
| **Emergency handling** | Generic disclaimer | Tiered (CALL NOW / URGENT), no negation suppression, LLM second-layer confirm |
| **Prescription reading** | None | Image quality gate → model reading → confidence level → pharmacist reminder |
| **Post-discharge support** | None | Medication reconciliation with Beers/interaction/duplicate flags |
| **Doctor communication** | None | SOAP-lite handoff summary the patient prints or shows on phone |
| **Accessibility** | Afterthought | TTS on every response, Simple Mode, navigation-first environment descriptions |
| **Reliability** | Single-model = single point of failure | 3-tier: Gemini → Groq → static safe response |
| **Privacy/consent** | Rarely addressed | Informed consent gate, session-only storage |

### Key Technical Decisions

1. **Tiered emergency detection, no negation suppression**: Tier 1 (CALL NOW) uses specific, unambiguous phrases. Tier 2 (URGENT) uses broader phrases + a second-layer LLM yes/no confirm to reduce alert fatigue. Crisis/self-harm phrases have a separate sub-tier that is *never* downgraded by LLM. No negation suppression — "seene mein dard nahi hai" will trigger tier-2 (intentional: false negatives cost lives). Generic phrases like "help me" and "my heart" removed from both tiers.

2. **Image quality is a soft-block**: Blur, glare and size checks run before the model is called. Rather than refusing, the quality note is prepended to the prompt so the model reads what it can and marks confidence LOW. Blind users cannot see blur — never hard-block them.

3. **Task-scoped context**: Each API call receives only the profile fields relevant to that task (e.g., vision gets `medicines` and `allergies` but not `region`). Reduces unnecessary PII sent to external APIs.

4. **Medication reconciliation at temperature 0 with JSON schema**: Structured JSON output parsed in Python and rendered as a typed Markdown table. Drug names copied verbatim — no auto-correction. Unclear timings (SOS, HS, PRN) flagged explicitly.

5. **Conversational memory**: Last 6 messages (3 exchanges) sent to the model on each symptom query. Follow-ups have context.

6. **Whisper STT fallback**: When `SpeechRecognition` fails on unsupported formats (webm from browser mic), Groq Whisper is tried. Language-specific BCP-47 code passed to Google STT. Hindi speech no longer defaults to US English.

7. **Results cached in `session_state`**: Vision results keyed by `(file_hash, task)` — switching tasks or uploading a new photo shows fresh results, not the previous one. Audio, handoff, and reconciliation results also cached.

---

## 🏗️ Architecture

```
User Input (Camera / Mic / Text)
        │
        ▼
 Consent Gate (one-time per session)
        │
        ▼
 Real-Time Context Builder
 { timestamp, region, language, profile, medicines, allergies }
        │
        ├──► Image Quality Check (blur / glare / size)
        │         └── Soft gate: quality note prepended, model reads with LOW confidence
        │              (never hard-block — blind users cannot see blur)
        │
        ▼
 Multilingual Emergency Detector (tiered phrases, 8 languages, token co-occurrence for snakebite)
         │
         ├── MATCH ──► 🚨 Hotlines panel (112/108 tel: links — crisis shows Tele-MANAS)
        │
        ▼
 LLM Orchestrator
  ├─ 1. Gemini 3.6 Flash — multimodal, streaming
  ├─ 2. Groq Qwen3.8-27b — text fallback
  └─ 3. Static safe response — final fallback
        │
        ▼
 Response → Markdown + TTS + Download
```

---

## ⚙️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Frontend | Streamlit | 1.35+ |
| Primary LLM | Google Gemini 3.6 Flash | `google-genai` SDK 2.29+ |
| Fallback LLM | Groq Qwen3.8-27b | `groq` SDK 0.9+ |
| Vision + image quality | Gemini 3.6 Flash + Pillow | PIL `ImageFilter`, `ImageStat` |
| In-browser mic | streamlit-mic-recorder | 0.0.8+ |
| STT | SpeechRecognition (Google free) | 3.10+ |
| TTS | Web Speech API | Browser-native |
| Python | **3.12** | 3.13/3.14 not supported by all deps |

---

## ⚠️ Trust, Safety & Known Limitations

This section exists because a health tool must be honest about what it cannot do.

### Emergency Detection
- Tier 1 (CALL NOW) uses specific, unambiguous phrases and fires **immediately without any LLM veto**.
- Tier 2 (URGENT) adds a second-layer LLM yes/no confirm to reduce alert fatigue. **Only a clean "NO" downgrades; ambiguous or partial output keeps the alert.** Crisis and snakebite phrases are **never downgraded**.
- **No negation suppression by design.** "seene mein dard nahi" will trigger tier-2. A false negative in an emergency costs lives; a false positive costs one tap.
- Hindi and English have the most complete coverage. Bhojpuri, Awadhi, Bengali, Tamil, Telugu are seed lists — not full coverage. Do not present these as production-ready.
- Tier-2 LLM confirm adds ~1–2 seconds. Tier-1 fires immediately.
- Past-tense phrasing ("had a seizure last year") triggers a "Is this happening NOW?" confirmation widget — only **strong time-reference markers** (e.g., "last year", "years ago") activate this; weak phrasing like "had a" does not.
- **Before production**: run `python eval_emergency.py` (see `EVALUATION.md`) and supplement with native-speaker-reviewed test phrases.

### Prescription Reading Limitations
- The model reports a confidence level (High/Medium/Low) but this is **self-assessed by the LLM**, not independently verified.
- Drug names are plausibility-checked by the model against its training knowledge — there is no integration with CDSCO, RxNorm, or openFDA in this MVP.
- **Never adjust medication doses based solely on AI output.** Always confirm with a licensed pharmacist.
- The image quality gate catches blur/glare/size issues but not all failure modes (e.g., partial label, non-standard packaging).

### Clinical Reliability
- AuraVoice has **not been evaluated by clinicians** and has **not been tested with real patients**.
- LLM outputs can contain errors, omissions, or outdated information.
- Urgency classifications (🔴/🟡/🟢) are AI-generated estimates, not clinical triage.

### Accessibility Gaps (Roadmap Items)
- TTS uses the browser's Web Speech API — quality and language support vary by device/browser.
- The app has not been tested with screen readers (TalkBack, VoiceOver, NVDA).
- Camera framing for blind users lacks audio guidance cues (e.g., beeps when label is centred).
- No WhatsApp or IVR (phone call) channel — limits reach to smartphone users with data connectivity.
- No offline mode for areas with intermittent connectivity.

### Privacy & Data
- AuraVoice does not store any user data beyond the current browser session.
- **Inputs are sent to Google Gemini and Groq APIs.** These services have their own data retention and usage policies. On free tiers, inputs may be used for model improvement. See:
  - [Google Generative AI Terms](https://policies.google.com/terms)
  - [Groq Privacy Policy](https://groq.com/privacy-policy/)
- For a production deployment handling patient data, a paid API tier with a Data Processing Agreement (DPA) is required.

### Regulatory Alignment
- **Privacy & Data Minimisation**: Operates with purpose limitation and zero persistent storage beyond the active session; users explicitly consent prior to entering information.
- Aligned with **ABDM/ABHA** digital health identity principles — users are directed to create a health ID at abdm.gov.in.
- **Not a medical device** under the Medical Devices Rules 2017 (India) or FDA 510(k) (USA). No regulatory clearance has been sought.

### Hotline Numbers
> ⚠️ **Every hotline number must be independently verified against [mohfw.gov.in](https://mohfw.gov.in) before deployment.** Numbers change. The list in `app.py` (`INDIA_HOTLINES`) carries a "Last checked: Oct 2026" comment — update it each time you verify. Removed e-Sanjeevani (number unverified) and Jan Aushadhi (pharmacy helpline, not a clinical emergency line). iCall is labelled with its limited hours (Mon–Sat 8 am–10 pm) so it is not mistaken for a 24/7 line.

---

## 🛠️ Local Development Setup

### Prerequisites
- **Python 3.12** (not 3.13 or 3.14)
- Google Gemini API key (free): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Groq API key (free, optional): [console.groq.com/keys](https://console.groq.com/keys)

### Step 1: Clone & Create Venv

```cmd
git clone https://github.com/pgovila/AuraVoice-Multimodal-Health-Accessibility-Agent.git
cd "AuraVoice-Multimodal-Health-Accessibility-Agent"
```

> ⚠️ **Windows**: Specify Python 3.12 explicitly if you have multiple versions:
```cmd
C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3.12 -m venv venv && source venv/bin/activate
```

### Step 2: Install Dependencies

```cmd
pip install -r requirements.txt
```

### Step 3: Configure API Keys

**Windows (cmd):**
```cmd
copy ".streamlit\secrets.toml.example" ".streamlit\secrets.toml"
notepad ".streamlit\secrets.toml"
```

Replace placeholders with real keys:
```toml
GEMINI_API_KEY = "AIzaSy..."
GROQ_API_KEY   = "gsk_..."
```

### Step 4: Run

```cmd
streamlit run app.py
```

### Step 5: First Use

1. Accept the **consent screen** (notice shown once per session)
2. Enter API keys in sidebar → click **🔗 Connect APIs**
3. Fill in your profile → click **💾 Save Profile**
4. Use any of the five tabs

---

## ☁️ Deployment to Streamlit Community Cloud

🌐 **Live Application URL:** [https://auravoice-multimodal-health-accessibility-agent-7sc3pkcpirqquh.streamlit.app/](https://auravoice-multimodal-health-accessibility-agent-7sc3pkcpirqquh.streamlit.app/)

### Step 1: Push to GitHub

```cmd
git init
git add app.py requirements.txt .streamlit\config.toml .streamlit\secrets.toml.example README.md
echo .streamlit/secrets.toml >> .gitignore
git add .gitignore
git commit -m "AuraVoice v2 — trust + accessibility update"
git remote add origin https://github.com/pgovila/AuraVoice-Multimodal-Health-Accessibility-Agent.git
git push -u origin main
```

### Step 2: Deploy

1. [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo → `app.py`
2. **Advanced settings** → **Secrets**:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   GROQ_API_KEY   = "gsk_..."
   ```
3. Click **Deploy**

---

## 🤗 Deployment to Hugging Face Spaces

1. [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: Streamlit → CPU Basic
2. Settings → Variables and secrets → add `GEMINI_API_KEY` and `GROQ_API_KEY` as secrets
3. `git remote add hf https://huggingface.co/spaces/pgovila/AuraVoice-Multimodal-Health-Accessibility-Agent && git push hf main`

---

## 📁 Project Structure

```
auravoice/
├── app.py                          # Full application — 18 modules, ~2500 lines
├── emergency.py                    # Pure-Python tier-1/2/crisis classifier (no Streamlit)
├── eval_emergency.py               # Standalone evaluation harness (no API keys needed)
├── eval_emergency.csv              # Independent test set (75 EN/HI utterances)
├── requirements.txt                # Python 3.12 dependencies
├── tests/
│   ├── test_emergency.py           # 51 pytest cases: Tier 1/2, crisis, snakebite, regressions
│   └── test_reconciliation.py      # 6 pytest cases: shorthand parser, null-safety, spoken text
├── .streamlit/
│   ├── config.toml                 # Dark theme + server settings
│   ├── secrets.toml.example        # API key template — safe to commit
│   └── secrets.toml                # Your real keys — NEVER commit
├── .gitignore
└── README.md
```

---

## 🐛 Troubleshooting

> For venv setup, Python version, and file-copy issues see [`DEVELOPMENT.md`](DEVELOPMENT.md).

| Problem | Cause | Fix |
|---|---|---|
| Static fallback / 404 model error | Model name outdated | Change `AURAVOICE_GEMINI_MODEL` / `AURAVOICE_GROQ_MODEL` env vars; re-run Connect & Test |
| Keys not persisting across reruns | SDK objects not picklable | Fixed — keys stored as strings, re-created from `@st.cache_resource` |
| `google-generativeai` FutureWarning | Deprecated SDK | Migrated to `google-genai` SDK |
| Emergency fires on "I have no chest pain" | No negation suppression (intentional) | Tier-2 fires (false negative cost > false positive cost). LLM confirm step may downgrade non-crisis tier-2 to 0 if LLM clearly says NO. Tier-1 and crisis are **never** downgraded. |
| Hindi voice transcribed as gibberish | STT language defaulted to English | Fixed — language-aware BCP-47 code passed to Google STT |
| Hindi TTS is silent (no audio) | ASCII strip removed Devanagari | Fixed — `_clean_for_tts` now uses Unicode-range emoji regex, preserves Hindi |
| Audio upload (MP3/OGG) not transcribed | SpeechRecognition doesn't support these formats | Groq Whisper fallback triggered automatically if Groq key is set |
| Vision tab shows stale results on new photo | Cache keyed by session, not file | Fixed — vision cache keyed by `(file_hash, task)` |
| Qwen `<think>` blocks appear in output | Qwen reasoning traces in content | Fixed — stripped via `_strip_think_blocks()` |
| Prescription image rejected | Hard quality block | Fixed — quality is now a soft-block; model reads with LOW confidence |
| "Connect & Test" self-test available for Streamlit Cloud secrets path | Button only visible in dev expander | On Streamlit Cloud, keys load from `st.secrets` automatically — verify via sidebar model status badge |

---

## 🔒 Security & Privacy

- All processing is in-memory. No health data, images, or audio are stored by AuraVoice.
- API keys are server-side in `st.secrets` on Streamlit Cloud — never sent to browser.
- Context JSON contains only user-provided profile data — no device IDs, IPs, or biometrics.
- Free-tier Gemini and Groq APIs may use inputs for model training. Use a paid tier with DPA for production health deployments.

---

## 📋 Suggested Roadmap 

See [`EVALUATION.md`](EVALUATION.md) for the evaluation protocol you should run before making credibility claims.

1. **Run the evaluation** in `EVALUATION.md` — report numbers before presenting.
2. **Drug database**: Cross-check drug names against CDSCO / RxNorm / openFDA.
3. **WhatsApp / IVR**: Reach rural users without apps via Twilio or BSNL IVR.
4. **Screen reader testing**: TalkBack (Android), VoiceOver (iOS), NVDA (Windows).
5. **Audio framing cues**: Beep when label is centred in frame for blind users.
6. **Caregiver mode**: Shared schedule + missed-dose alerts.
7. **Offline mode**: Cached hotlines + drug info for low-connectivity.
8. **Native speaker review**: Hindi and English emergency phrases reviewed by medical professionals.

---

## ⚠️ Disclaimer

AuraVoice is an AI-assisted health information and accessibility tool. It does **not** provide medical diagnosis, prescriptions, or treatment. It has not been clinically evaluated. Always consult a qualified and licensed healthcare professional for medical decisions. In a medical emergency, call **112** (India: **108** for ambulance, **14416** for mental health).

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with ❤️ · AuraVoice Inclusive Health Assistant*
