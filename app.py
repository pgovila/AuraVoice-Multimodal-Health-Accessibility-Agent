"""
AuraVoice: Multimodal Health & Accessibility Agent
===================================================
Track: Inclusive Technology

Architecture
------------
- Frontend  : Streamlit 1.35+
- Primary   : Google Gemini 3.6 Flash (multimodal) via google-genai SDK
- Fallback  : Groq Qwen3.8-27b (text-only) via Groq SDK
- STT       : SpeechRecognition (language-aware) + streamlit-mic-recorder
- Safety    : Tiered multilingual emergency detection (no false-suppress) → LLM confirm
- Privacy   : No PII stored; task-scoped context; DPDP 2023 aligned

Module map
----------
 1. constants        – model IDs, emergency phrases, prompts, hotlines
 2. image_quality    – blur / glare / size checks (soft-block)
 3. emergency        – tiered detection, NO negation suppression, LLM confirm
 4. llm              – Gemini + Groq init + orchestration + streaming
 5. vision_agent     – image → analysis (quality-soft-gated)
 6. audio_agent      – mic bytes → language-aware STT or Whisper → LLM
 7. symptom_agent    – text symptoms → guided triage + conversational memory
 8. handoff_agent    – doctor handoff (verbatim patient words + summary)
 9. reconcile_agent  – discharge reconciliation (structured JSON → table)
10. session          – st.session_state initialisation
11. sidebar          – API keys, profile, simple-mode toggle
12. tts              – text-to-speech (markdown-stripped, safe JS)
13. ui_tabs          – all tab renderers (results cached in session_state)
14. main             – entry point (startup self-test, secrets guard)
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import hashlib
import html as html_lib
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── emergency module (pure Python — no Streamlit, safe to import everywhere) ──
from emergency import (
    classify as _em_classify,
    Decision as EmDecision,
    is_strong_past_tense,
    is_devanagari,
    SNAKEBITE_FIRST_AID,
)

# ── third-party ───────────────────────────────────────────────────────────────
import streamlit as st
from PIL import Image, ImageFilter, ImageStat

# ── optional deps (graceful degradation) ─────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("auravoice")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Make model IDs configurable via env so they can be updated without code change
GEMINI_MODEL = os.getenv("AURAVOICE_GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL   = os.getenv("AURAVOICE_GROQ_MODEL",   "qwen/qwen3.8-27b")

# STT language codes per user-profile language (BCP-47)
STT_LANG_CODES = {
    "English":    "en-IN",
    "Hindi":      "hi-IN",
    "Tamil":      "ta-IN",
    "Telugu":     "te-IN",
    "Bengali":    "bn-IN",
    "Spanish":    "es-ES",
    "French":     "fr-FR",
    "Arabic":     "ar-SA",
    "Mandarin":   "zh-CN",
    "Portuguese": "pt-BR",
    "Swahili":    "sw-KE",
    "Bhojpuri":   "hi-IN",   # closest supported code
    "Awadhi":     "hi-IN",
    "Punjabi":    "pa-IN",
}

# TTS locale codes per language (Web Speech API BCP-47)
TTS_LANG_CODES = {
    "English":    "en-IN",
    "Hindi":      "hi-IN",
    "Tamil":      "ta-IN",
    "Telugu":     "te-IN",
    "Bengali":    "bn-IN",
    "Spanish":    "es-ES",
    "French":     "fr-FR",
    "Arabic":     "ar-SA",
    "Mandarin":   "zh-CN",
    "Portuguese": "pt-BR",
    "Swahili":    "sw-KE",
    "Bhojpuri":   "hi-IN",
    "Awadhi":     "hi-IN",
    "Punjabi":    "pa-IN",
}

# ---------------------------------------------------------------------------
# TIERED EMERGENCY PHRASES
# Tier 1 = CALL NOW (unambiguous life-threat)
# Tier 2 = URGENT (serious, see doctor today)
# NO negation suppression — a false negative here costs lives.
# Generic phrases ("help me", "emergency", "my heart") are tier 2 or removed.
# ---------------------------------------------------------------------------
TIER1_PHRASES: list[tuple[str, str]] = [
    # English – cardiac arrest / MI
    ("heart attack", "en"), ("cardiac arrest", "en"), ("crushing chest pain", "en"),
    ("chest pain radiating", "en"), ("pain in left arm", "en"),
    # English – respiratory arrest
    ("not breathing", "en"), ("stopped breathing", "en"),
    ("cannot breathe at all", "en"), ("airway blocked", "en"), ("choking", "en"),
    # English – stroke (FAST signs)
    ("face drooping", "en"), ("arm weakness suddenly", "en"), ("speech slurred", "en"),
    ("sudden vision loss", "en"), ("worst headache of my life", "en"),
    # English – severe bleeding / shock
    ("bleeding won't stop", "en"), ("bleeding profusely", "en"),
    ("unconscious", "en"), ("unresponsive", "en"), ("not waking up", "en"),
    ("no pulse", "en"),
    # English – poisoning / overdose
    ("overdose", "en"), ("swallowed poison", "en"), ("took too many pills", "en"),
    # English – seizure ongoing
    ("seizure won't stop", "en"), ("convulsions", "en"),
    # English – mental health (active crisis)
    ("going to kill myself", "en"), ("about to end my life", "en"),
    ("taking pills to die", "en"),
    # English – obstetric emergency
    ("eclampsia", "en"), ("baby not moving for hours", "en"),
    ("heavy bleeding in pregnancy", "en"),
    # English – envenomation
    ("bitten by snake", "en"), ("snakebite", "en"), ("scorpion sting severe", "en"),
    # Hindi – cardiac
    ("dil ka daura pad gaya", "hi"), ("dil ka dora", "hi"),
    ("seene mein bahut tez dard", "hi"), ("chhati mein tez dard", "hi"),
    ("seene mein dard aa raha hai", "hi"), ("दिल का दौरा", "hi"),
    ("सीने में बहुत तेज दर्द", "hi"),
    # Hindi – respiratory
    ("saans bilkul nahi aa rahi", "hi"), ("dam ghut gaya", "hi"),
    ("dam ghut raha hai", "hi"), ("दम घुट रहा है", "hi"),
    ("sans band ho gayi", "hi"),
    # Hindi – stroke
    ("laqwa pad gaya", "hi"), ("muh tedha ho gaya", "hi"),
    ("haath achanak kaam nahi kar raha", "hi"), ("bolna band ho gaya", "hi"),
    # Hindi – bleeding
    ("khoon bahut bahut aa raha hai", "hi"), ("khoon band nahi ho raha", "hi"),
    ("behosh ho gaya", "hi"), ("होश नहीं है", "hi"),
    # Hindi – mental health
    ("marne ki koshish kar raha hoon", "hi"), ("jaan de doonga", "hi"),
    ("khud ko maar loonga", "hi"),
    # Hindi – obstetric
    ("bachcha hil nahi raha kal se", "hi"), ("pani toot gaya", "hi"),
    ("prasav mein bahut khoon", "hi"),
    # Hindi – snakebite
    ("saanp ne kata abhi", "hi"), ("bichchu ne kata", "hi"),
    # Bhojpuri
    ("seena mein bahut dard ba", "bh"), ("dam bilkul naikhe", "bh"),
    ("behosh ba", "bh"), ("saanp katlis", "bh"), ("marab chahta hai", "bh"),
    # Awadhi
    ("sina mein bahut dard", "aw"), ("dam naahin aawat", "aw"),
    # Bengali
    ("বুকে অনেক ব্যথা", "bn"), ("শ্বাস নিতে পারছি না একদম", "bn"),
    ("bukhe onek byatha", "bn"),
    # Tamil
    ("நெஞ்சில் மிகவும் வலி", "ta"), ("மூச்சே வரவில்லை", "ta"),
    # Telugu
    ("ఛాతీలో చాలా నొప్పి", "te"), ("శ్వాస రావడం లేదు", "te"),
]

TIER2_PHRASES: list[tuple[str, str]] = [
    # These warrant urgent advice but not a 112 call immediately
    ("chest pain", "en"), ("chest tightness", "en"),
    ("shortness of breath", "en"), ("difficulty breathing", "en"),
    ("severe headache", "en"), ("sudden confusion", "en"),
    ("seizure", "en"), ("fitting", "en"),
    ("severe bleeding", "en"), ("poisoning", "en"),
    ("want to die", "en"), ("kill myself", "en"), ("suicide", "en"),
    ("self harm", "en"), ("no reason to live", "en"),
    ("labour pain", "en"), ("water broke", "en"),
    ("snake bite", "en"), ("scorpion sting", "en"),
    ("seene mein dard", "hi"), ("सीने में दर्द", "hi"),
    ("saans nahi aa rahi", "hi"), ("saans lene mein takleef", "hi"),
    ("behosh", "hi"), ("hosh nahi", "hi"),
    ("marna chahta hoon", "hi"), ("marna chahti hoon", "hi"),
    ("jaan dena chahta", "hi"),
    ("saanp ne kata", "hi"), ("saanp ka dank", "hi"),
    ("prasav peeda", "hi"), ("bachcha hil nahi raha", "hi"),
    ("seena mein dard ba", "bh"), ("dam naikhe", "bh"),
    ("bukhe byatha", "bn"), ("shash nite koshto", "bn"),
    ("நெஞ்சு வலி", "ta"), ("மூச்சு சிரமம்", "ta"),
    ("ఛాతీ నొప్పి", "te"), ("శ్వాస తీసుకోవడం కష్టం", "te"),
]

# Verified India hotlines — verify against MoHFW / NHA before deployment
# Last checked against publicly available sources: Oct 2026
INDIA_HOTLINES = {
    "Emergency (Police/Fire/Medical)": "112",
    "Ambulance (EMRI)":                "108",
    "Maternal & Child Ambulance":      "102",
    "Tele-MANAS (Mental Health 24/7)": "14416",
    "iCall (Mon–Sat 8 am–10 pm only)": "9152987821",
    "Childline":                       "1098",
    "Elderline":                       "14567",
    "Poison Control (AIIMS Delhi)":    "011-26593677",
}
# ⚠️  Always verify each number on MoHFW.gov.in before publishing or presenting.
# Removed: e-Sanjeevani (number unverified), Jan Aushadhi (pharmacy helpline, not clinical).

GLOBAL_HOTLINES = {
    "India Emergency":  "112",
    "USA":              "911",
    "UK":               "999",
    "Australia":        "000",
    "Europe":           "112",
}

SAFE_FALLBACK_RESPONSE = (
    "⚠️ I'm having trouble connecting right now.\n\n"
    "Please consult a qualified healthcare professional.\n\n"
    "If this is an emergency, call **112** (India: **108** for ambulance, "
    "**14416** for mental health crisis).\n\n"
    "⚠️ This is not a medical diagnosis. Please consult a qualified doctor."
)

SYSTEM_PROMPT = """You are AuraVoice, a compassionate, multilingual health literacy assistant.

Your users are elderly individuals, visually impaired people, low-literacy patients,
and people in underserved areas of India. Many speak Hindi or regional languages.
Respond in the same language the user writes or speaks.

CORE RULES:
1. You provide health INFORMATION and LITERACY support — not diagnosis or prescription.
2. Every response MUST end with: "⚠️ This is not a medical diagnosis. Please consult a qualified doctor."
3. For symptoms: give urgency level (🔴 Urgent — go now / 🟡 Soon — within 3 days / 🟢 Routine).
   Always BIAS TOWARDS HIGHER URGENCY for: age under 5, over 65, pregnancy, anticoagulant use.
4. For prescriptions: report confidence HIGH / MEDIUM / LOW. If LOW, say "ask your pharmacist".
5. TEACH-BACK: after any medication schedule, ask: "To make sure I explained clearly — can you tell me when you will take this medicine?"
6. TONE: warm, calm, numbered steps, plain language (assume Class 8 literacy).
7. CONVERSATIONAL MEMORY: Previous messages are included. Use them for follow-up context.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE QUALITY (SOFT-BLOCK — read anyway, mark confidence low)
# ─────────────────────────────────────────────────────────────────────────────

def check_image_quality(img: Image.Image) -> dict:
    """
    Soft quality check using Laplacian edge-variance (FIND_EDGES filter),
    brightness, and minimum dimensions.
    Returns {ok, issues, guidance, sharpness, brightness} — never hard-blocks.
    """
    issues = []
    gray = img.convert("L")

    # Sharpness via edge-map variance (Laplacian proxy)
    edges     = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).var[0]
    if sharpness < 25:
        issues.append("blurry")

    # Brightness
    brightness = ImageStat.Stat(gray).mean[0]
    if brightness < 40:
        issues.append("too_dark")
    elif brightness > 215:
        issues.append("overexposed")

    # Size
    if img.width < 180 or img.height < 180:
        issues.append("too_small")

    guidance_map = {
        "blurry":      "📸 Hold the camera steady and wait for it to focus before capturing.",
        "too_dark":    "💡 Move to better lighting or switch on a lamp.",
        "overexposed": "↩️ Tilt the label slightly to reduce glare from light.",
        "too_small":   "🔍 Move closer so the text fills most of the frame.",
    }
    guidance = "  ".join(guidance_map[i] for i in issues)
    return {
        "ok":         len(issues) == 0,
        "issues":     issues,
        "guidance":   guidance,
        "sharpness":  round(sharpness, 1),
        "brightness": round(brightness, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMERGENCY DETECTION — thin wrapper around emergency.py
# All detection logic lives in emergency.py (pure Python, testable with pytest).
# ─────────────────────────────────────────────────────────────────────────────

# Languages where emergency phrase coverage is incomplete
_LANGUAGES_WITH_NO_EMERGENCY_COVERAGE = frozenset(
    ["Spanish", "French", "Arabic", "Mandarin", "Portuguese", "Swahili"]
)

# Emergency numbers by region (for crisis copy)
_REGION_EMERGENCY_NUMBERS = {
    "India":     ("112", "108"),
    "USA":       ("911", "911"),
    "UK":        ("999", "999"),
    "Australia": ("000", "000"),
    "Global":    ("112", "112"),
}

# Crisis warm-line copy (region-specific helplines)
def _crisis_copy(region: str = "India") -> str:
    if region == "India":
        return (
            "You are not alone. A counsellor is ready to talk right now.\n\n"
            "📞 [**Tele-MANAS: 14416**](tel:14416) (Govt of India, free, 24/7, Hindi & English)\n"
            "📞 [**iCall: 9152987821**](tel:9152987821) (TISS, Mon–Sat, 8 am–10 pm)\n\n"
            "If you are in immediate physical danger, call 📞 [**112**](tel:112)."
        )
    elif region == "USA":
        return (
            "You are not alone. Support is available right now.\n\n"
            "📞 [**988 Suicide & Crisis Lifeline: 988**](tel:988) (free, 24/7, call or text)\n"
            "💬 Crisis Text Line: Text **HOME** to **741741**\n\n"
            "If you are in immediate danger, call 📞 [**911**](tel:911)."
        )
    elif region == "UK":
        return (
            "You are not alone. Free confidential help is available 24/7.\n\n"
            "📞 [**Samaritans: 116 123**](tel:116123) (free, 24/7)\n"
            "📞 [**NHS Mental Health: 111**](tel:111) (free, 24/7)\n\n"
            "If you are in immediate danger, call 📞 [**999**](tel:999)."
        )
    elif region == "Australia":
        return (
            "You are not alone. Support is available 24/7.\n\n"
            "📞 [**Lifeline: 13 11 14**](tel:131114) (free, 24/7 crisis support)\n"
            "📞 [**Beyond Blue: 1300 22 4636**](tel:1300224636) (free, 24/7)\n\n"
            "If you are in immediate danger, call 📞 [**000**](tel:000)."
        )
    em, _ = _REGION_EMERGENCY_NUMBERS.get(region, ("112", "112"))
    return (
        "You are not alone. Free and confidential support is available.\n\n"
        "🌐 International crisis resources: [findahelpline.com](https://findahelpline.com)\n\n"
        f"If you are in immediate danger, call 📞 [**{em}**](tel:{em})."
    )


def render_emergency_alert(
    dec: "EmDecision",
    region: str = "India",
) -> None:
    """
    Render tiered emergency banner from a Decision object.
    - Snakebite (dec.snakebite): red + WHO first-aid, no LLM.
    - Crisis (dec.crisis): warm blue Tele-MANAS / regional crisis copy.
    - Tier 1: red call-now (112/108 tel: links only).
    - Tier 2: orange urgent.
    - Info query (dec.info_query): shows a softer banner + "Not an emergency" escape.
    Caller must also store the alert in chat_history for rerun persistence.
    """
    em_num, amb_num = _REGION_EMERGENCY_NUMBERS.get(region, ("112", "112"))

    if dec.snakebite:
        st.error(SNAKEBITE_FIRST_AID, icon="🐍")
        return

    if dec.crisis:
        st.error(
            f"💙 **We see you are going through something very hard.**\n\n"
            f"{_crisis_copy(region)}",
            icon="💙",
        )
        return

    if dec.info_query:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.info(
                f"ℹ️ **Informational query noted** — Medical explanation provided below.\n\n"
                f"If this is a real acute event happening right now, call 📞 [**{em_num}**](tel:{em_num}) immediately.",
                icon="ℹ️",
            )
        with c2:
            if st.button("Dismiss\n(Not an emergency)", key=f"dismiss_info_{abs(hash(dec.matched)) % 100000}", use_container_width=True):
                st.session_state["chat_history"] = [
                    m for m in st.session_state.get("chat_history", [])
                    if not (m.get("role") == "alert" and m.get("info_query"))
                ]
                st.rerun()
        return

    if dec.tier == 1:
        if region == "India":
            st.error(
                "🚨 **CALL EMERGENCY SERVICES NOW — Do not wait for AI guidance.**\n\n"
                f"📞 [**Call {em_num}**](tel:{em_num})  ·  "
                f"🚑 [**Call {amb_num} (Ambulance)**](tel:{amb_num})",
                icon="🚨",
            )
        else:
            st.error(
                f"🚨 **CALL EMERGENCY SERVICES NOW**\n\n"
                f"📞 [**{em_num}**](tel:{em_num})",
                icon="🚨",
            )
    elif dec.tier == 2:
        st.warning(
            "⚠️ **URGENT — Please seek medical help today.**\n\n"
            f"If symptoms worsen, call 📞 [**{em_num}**](tel:{em_num})  ·  "
            f"🚑 [**{amb_num}**](tel:{amb_num})\n\n"
            "Mental health crisis → **14416** (Tele-MANAS, 24/7)",
            icon="⚠️",
        )


def llm_emergency_confirm(text: str, gemini_client, groq_client) -> bool:
    """
    Second-layer LLM yes/no check — ONLY used for tier-2 non-crisis, non-snakebite.
    NEVER called for tier-1, crisis, or snakebite (those are never downgraded).

    Safety rule: ONLY a clean "NO" downgrades. Anything else = fail-safe True (urgent).
    Uses 64 output tokens so thinking models have room to output the word.
    """
    prompt = (
        "Does the following text describe a medical emergency or urgent health crisis "
        "requiring immediate medical attention TODAY?\n\n"
        f"Text: \"{text}\"\n\n"
        "Reply with ONLY the single word YES or NO. No other text."
    )
    for client, call_fn in [(gemini_client, _call_gemini_raw), (groq_client, _call_groq_raw)]:
        if client is not None:
            raw = call_fn(client, prompt)
            if raw:
                cleaned = _strip_think_blocks(raw).strip().strip(".").lower()
                if cleaned == "no":
                    return False
                return True   # "yes", partial, garbled → fail-safe
    return True  # LLM unavailable → assume urgent


def _call_gemini_raw(client, prompt: str) -> Optional[str]:
    """Minimal Gemini call — no system prompt, no streaming. 64 tokens with thinking budget 0."""
    try:
        config_kwargs: dict = {
            "temperature": 0.0,
            "max_output_tokens": 64,
        }
        if hasattr(genai_types, "ThinkingConfig"):
            try:
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        return resp.text
    except Exception:
        return None


def _call_groq_raw(client, prompt: str) -> Optional[str]:
    """Minimal Groq call — no system prompt. 64 tokens."""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def init_gemini(api_key: str):
    """Initialise google-genai Client. Cached by key string."""
    if not GEMINI_AVAILABLE or not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        log.info("Gemini client initialised ✓")
        return client
    except Exception as exc:
        log.warning("Gemini init failed: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def init_groq(api_key: str):
    """Initialise Groq client. Cached by key string."""
    if not GROQ_AVAILABLE or not api_key:
        return None
    try:
        client = Groq(api_key=api_key)
        log.info("Groq client initialised ✓")
        return client
    except Exception as exc:
        log.warning("Groq init failed: %s", exc)
        return None


def startup_self_test(gemini_client, groq_client) -> dict:
    """
    Quick connectivity test run once after API keys are set.
    Returns {gemini_ok, groq_ok, gemini_model, groq_model}.
    """
    results = {"gemini_ok": False, "groq_ok": False}
    if gemini_client:
        resp = _call_gemini_raw(gemini_client, "Reply with the single word: OK")
        results["gemini_ok"] = resp is not None and len(resp.strip()) < 20
    if groq_client:
        resp = _call_groq_raw(groq_client, "Reply with the single word: OK")
        results["groq_ok"] = resp is not None and len(resp.strip()) < 20
    return results


# Standard IANA timezones supporting Daylight Saving Time (DST)
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo  # type: ignore

_REGION_TIMEZONES = {
    "India":     "Asia/Kolkata",
    "Global":    "UTC",
    "USA":       "America/New_York",
    "UK":        "Europe/London",
    "Australia": "Australia/Sydney",
}

def _get_local_time_for_region(region: str) -> datetime:
    tz_name = _REGION_TIMEZONES.get(region, "Asia/Kolkata")
    try:
        return datetime.now(zoneinfo.ZoneInfo(tz_name))
    except Exception:
        return datetime.now(timezone.utc)


def build_context(user_profile: dict, task_fields: list[str]) -> str:
    """
    Build minimal task-scoped context — only include profile fields
    relevant to the current task to avoid sending unnecessary PII to the API.
    task_fields: subset of ['age','region','language','conditions','medicines','allergies']
    NOTE: 'name' is intentionally excluded — no clinical value, reduces PII footprint.
    NOTE: 'region' is included ONLY when explicitly in task_fields (e.g. vision excludes it).
    """
    region = user_profile.get("region", "India")
    local_time = _get_local_time_for_region(region)

    ctx: dict = {
        "local_time":  local_time.strftime("%H:%M"),
        "day_of_week": local_time.strftime("%A"),
    }
    # 'name' is deliberately excluded from all task contexts
    # 'region' is only included when the caller explicitly requests it in task_fields
    allowed = {"age", "region", "language", "conditions", "medicines", "allergies"}
    for field in task_fields:
        if field == "name":
            continue   # never send name — no clinical value
        if field not in allowed:
            continue
        val = user_profile.get(field)
        if val:
            ctx[field] = val
    return json.dumps(ctx, indent=2, ensure_ascii=False)


def inject_context(user_message: str, context_json: str) -> str:
    return (
        f"{user_message}\n\n"
        f"---\n[Context — use to personalise, do not repeat verbatim]\n"
        f"```json\n{context_json}\n```"
    )


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen models sometimes emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_gemini(
    client,
    prompt: str,
    image: Optional[Image.Image] = None,
    retries: int = 2,
    stream: bool = False,
    _stream_placeholder=None,
) -> Optional[str]:
    """
    Call Gemini. When stream=True the caller must pass a `_stream_placeholder`
    (an st.empty() created INSIDE a `with st.chat_message("assistant"):` block)
    so streamed text renders inside the chat bubble, not outside it.
    """
    contents = [prompt] if image is None else [image, prompt]
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
        # 2048 output tokens; thinking tokens consume budget first on Flash-thinking
        # Use non-thinking alias when available via env override
        max_output_tokens=2048,
        top_p=0.9,
    )
    for attempt in range(retries + 1):
        try:
            if stream:
                placeholder = _stream_placeholder if _stream_placeholder is not None else st.empty()
                full_text = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL, contents=contents, config=config
                ):
                    if chunk.text:
                        full_text += chunk.text
                        placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
                return full_text or None
            else:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents, config=config
                )
                return resp.text or None
        except Exception as exc:
            err = str(exc).lower()
            if ("quota" in err or "rate" in err or "429" in err) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            log.error("Gemini failed (attempt %d): %s", attempt, exc)
            st.session_state["last_error"] = "Gemini error (check terminal logs)"
            return None
    return None


def call_groq(
    client,
    prompt: str,
    retries: int = 2,
    system: Optional[str] = None,
) -> Optional[str]:
    """
    Call Groq Qwen. Pass system=<str> to override the default SYSTEM_PROMPT,
    e.g. system='Output ONLY valid JSON' for reconciliation.
    """
    sys_msg = system if system is not None else SYSTEM_PROMPT
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            raw = completion.choices[0].message.content or ""
            return _strip_think_blocks(raw) or None
        except Exception as exc:
            err = str(exc).lower()
            if ("rate" in err or "429" in err) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            log.error("Groq failed (attempt %d): %s", attempt, exc)
            st.session_state["last_error"] = "Groq error (check terminal logs)"
            return None
    return None


def run_agent(
    prompt: str,
    image: Optional[Image.Image],
    gemini_client,
    groq_client,
    context_json: str,
    stream: bool = False,
    stream_placeholder=None,
) -> str:
    """
    Gemini → Groq → static fallback. Updates active_model badge.

    stream_placeholder: an st.empty() created inside a chat_message("assistant") block.
      Pass this when stream=True so Gemini streams inside the bubble.
      Groq does NOT stream — when Groq is used the caller renders the returned string.
      This function NEVER calls st.markdown() for Groq results; that is the caller's job.
    """
    enriched = inject_context(prompt, context_json)

    if gemini_client is not None:
        st.session_state["active_model"] = GEMINI_MODEL
        result = call_gemini(
            gemini_client, enriched, image,
            stream=stream, _stream_placeholder=stream_placeholder
        )
        if result:
            return result

    if groq_client is not None:
        st.session_state["active_model"] = f"groq/{GROQ_MODEL}"
        note = "[Image analysis unavailable in text-fallback mode]\n\n" if image else ""
        if stream:
            with st.spinner("Connecting to fallback model…"):
                result = call_groq(groq_client, enriched)
        else:
            result = call_groq(groq_client, enriched)
        if result:
            # Return text ONLY — caller decides when/how to render
            return note + result

    st.session_state["active_model"] = "static-fallback"
    return SAFE_FALLBACK_RESPONSE


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISION AGENT
# ─────────────────────────────────────────────────────────────────────────────

VISION_PROMPTS = {
    "prescription": (
        "You are reading a medicine label or prescription strip.\n"
        "Step 1: Transcribe EVERY visible word VERBATIM — drug name, strength, dose, "
        "frequency, manufacturer, expiry date, batch number, warnings.\n"
        "Step 2: Explain each field in simple plain language (Class 8 literacy).\n"
        "Step 3: State confidence — HIGH (all text clear), MEDIUM (some unclear), "
        "LOW (mostly unreadable — advise pharmacist visit).\n"
        "Step 4: Flag warnings (alcohol, driving, allergy alerts).\n"
        "Step 5: If a drug name looks like a possible OCR misread, note it as "
        "'Possible read: X — please verify with pharmacist'. "
        "Do NOT auto-correct drug names — look-alike drugs are dangerous.\n"
        "Step 6: End with: 'Please confirm all details with your pharmacist before taking.'\n"
        "Step 7: Teach-back: 'Can you tell me when you will take this medicine?'"
    ),
    "environment": (
        "Describe this space for a blind or visually impaired person navigating it.\n"
        "Use clock positions (12 o'clock = straight ahead) and approximate step counts.\n"
        "Focus only on navigation-critical information: obstacles, distances, floor hazards, "
        "exits, steps, ramps, signs.\n"
        "Do NOT describe colours, aesthetics, or decorations.\n"
        "⚠️ NOTE: Distances from a single photo are estimates — the user should proceed carefully "
        "and not rely solely on this description near stairs, traffic or uneven ground."
    ),
    "symptoms_photo": (
        "A patient has shared an image of a physical symptom.\n"
        "1. Describe what you observe objectively.\n"
        "2. Note that photo-based assessment is unreliable, especially across skin tones.\n"
        "3. Give urgency: 🔴 Urgent / 🟡 Soon / 🟢 Routine — but bias HIGH.\n"
        "4. Recommend seeing a doctor. Do NOT give a diagnosis.\n"
        "5. End with: 'This is not a medical diagnosis. Please see a doctor.'"
    ),
    "document": (
        "Transcribe ALL text in this medical document completely and accurately.\n"
        "Then: plain-language summary of what it says and what the patient needs to do.\n"
        "Highlight abnormal values with plain explanations.\n"
        "End with 3 questions to ask their doctor about this report."
    ),
    "discharge": (
        "This is a hospital discharge summary or medication instruction sheet.\n"
        "1. List every medicine: name, dose, timing, duration.\n"
        "2. Create a simple daily schedule table (Morning / Afternoon / Evening / Night).\n"
        "3. Flag interactions (e.g., aspirin + warfarin).\n"
        "4. Flag duplicates (same active ingredient twice).\n"
        "5. Flag Beers Criteria risks for patients over 65 (benzodiazepines, anticholinergics).\n"
        "6. List follow-up appointments.\n"
        "7. List warning signs to return to hospital immediately.\n"
        "8. Suggest Jan Aushadhi generics where applicable.\n"
        "9. End: 'Confirm this schedule with your pharmacist or doctor.'\n"
        "10. Teach-back: 'Can you tell me when you will take each medicine?'"
    ),
}


def process_camera_input(
    image_file,
    task: str,
    gemini_client,
    groq_client,
    user_profile: dict,
) -> tuple[str, dict]:
    """
    image bytes → quality check (soft) → LLM.
    Returns (response_text, quality_report).
    Quality issues are shown as a warning but do NOT block — blind users cannot
    see blur so we always attempt the read.
    """
    try:
        img = Image.open(image_file).convert("RGB")
        if max(img.size) > 1280:
            ratio = 1280 / max(img.size)
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS
            )

        quality = check_image_quality(img)
        ctx = build_context(user_profile, ["age", "language", "medicines", "allergies"])
        prompt = VISION_PROMPTS.get(task, VISION_PROMPTS["environment"])

        # Soft-block: prepend quality note to prompt rather than refusing
        if not quality["ok"]:
            quality_note = (
                f"Image quality issues detected: {', '.join(quality['issues'])}. "
                f"Try to read what you can, but state confidence as LOW and advise the user "
                f"to retake the photo with: {quality['guidance']}"
            )
            prompt = quality_note + "\n\n" + prompt

        response = run_agent(prompt, img, gemini_client, groq_client, ctx)
        return response, quality

    except Exception as exc:
        log.error("Vision processing error: %s", exc)
        return f"⚠️ Could not process image. Please try again.", {}


# ─────────────────────────────────────────────────────────────────────────────
# 6. AUDIO AGENT — language-aware STT + Whisper fallback path
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, language: str = "English") -> str:
    """
    Transcribe audio bytes with language-aware Google STT.
    Accepts WAV, AIFF, FLAC. For other formats (mp3, ogg, webm) from mic recorder,
    SpeechRecognition will raise AudioFile error — we catch and note it.
    Production upgrade: use Groq Whisper for multilingual + format support.
    """
    if not SR_AVAILABLE:
        return "[SpeechRecognition not installed]"

    lang_code = STT_LANG_CODES.get(language, "en-IN")
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language=lang_code)
    except sr.UnknownValueError:
        return "[Could not understand audio — please speak clearly and try again]"
    except sr.RequestError as exc:
        return f"[STT service error: {exc}]"
    except Exception as exc:
        # Likely unsupported format (webm from browser mic recorder)
        return f"[Audio format not supported for transcription: {type(exc).__name__}. Please type your query below.]"


def _detect_audio_format(header: bytes) -> str:
    """Return a file extension hint from magic bytes for Whisper naming."""
    if header[:4] == b"RIFF":
        return "wav"
    if header[:3] == b"ID3" or header[:2] == b"\xff\xfb":
        return "mp3"
    if header[:4] == b"OggS":
        return "ogg"
    if header[:4] == b"\x1aE\xdf\xa3":
        return "webm"   # MKV/WebM
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "m4a"    # MP4 / M4A (Safari iOS/macOS audio)
    return "webm"   # safe default for browser recordings


def transcribe_with_whisper(audio_bytes: bytes, groq_client, language: str = "hi") -> str:
    """
    Whisper via Groq — handles mp3/webm/ogg/m4a/wav.
    Uses magic-byte detection to set the correct filename extension.
    """
    if groq_client is None:
        return ""
    try:
        ext = _detect_audio_format(audio_bytes[:16])
        buf = io.BytesIO(audio_bytes)
        buf.name = f"audio.{ext}"
        resp = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=buf,
            language=language[:2].lower(),   # ISO 639-1 code
        )
        return resp.text
    except Exception as exc:
        log.warning("Whisper transcription failed: %s", exc)
        return ""


def process_audio_input(
    audio_bytes: bytes,
    gemini_client,
    groq_client,
    user_profile: dict,
) -> tuple[str, str, "EmDecision"]:
    """
    bytes → STT (with Whisper fallback) → emergency classify → LLM.
    Returns (transcript, response, decision) so caller can render alert + cache it.

    If Devanagari text is detected in the transcript, uses the LLM to classify
    symptoms rather than relying on low-recall Devanagari phrase matching.
    """
    language = user_profile.get("language", "English")
    transcript = transcribe_audio(audio_bytes, language)

    # If native STT failed on format, try Whisper
    if transcript.startswith("[Audio format") or transcript.startswith("[STT service"):
        lang_code = STT_LANG_CODES.get(language, "hi-IN")
        whisper_result = transcribe_with_whisper(audio_bytes, groq_client, lang_code)
        if whisper_result:
            transcript = whisper_result

    if transcript.startswith("["):
        from emergency import Decision
        return transcript, SAFE_FALLBACK_RESPONSE, Decision()

    dec = detect_symptom_emergency(transcript, gemini_client, groq_client, user_profile)

    # Fold crisis into tier (minimum tier 2)
    if dec.crisis:
        dec.tier = max(dec.tier, 2)

    region = user_profile.get("region", "India")
    if dec.snakebite:
        return transcript, SNAKEBITE_FIRST_AID, dec
    if dec.crisis:
        canned = (
            "💙 **Please know you are not alone.**\n\n"
            f"{_crisis_copy(region)}\n\n"
            "AuraVoice is here with you. Please reach out to a counsellor right now."
        )
        return transcript, canned, dec
    if dec.tier == 1 and not dec.info_query:
        canned = (
            "🚨 **EMERGENCY DETECTED. Call 📞 112 / 🚑 108 immediately.**\n\n"
            "Do not wait for AI guidance."
        )
        return transcript, canned, dec

    response = generate_symptom_response(
        transcript, gemini_client, groq_client, user_profile, [], dec, stream=False
    )
    return transcript, response, dec


# ─────────────────────────────────────────────────────────────────────────────
# 7. SYMPTOM CHECKER — guided triage + conversational memory
# ─────────────────────────────────────────────────────────────────────────────

_ANTICOAG_ROOTS = frozenset({
    "warfarin", "aspirin", "clopidogrel", "heparin", "rivaroxaban",
    "apixaban", "dabigatran", "acenocoumarol", "acitrom", "edoxaban",
})

# Keywords that bias the anticoagulant urgency note (only for bleeding/falls/head injury/surgery)
_ANTICOAG_CONTEXT_WORDS = frozenset({
    "bleed", "bleeding", "blood", "bruise", "bruising", "cut", "wound",
    "fall", "fell", "hit", "surgery", "operation", "procedure", "injury",
    "head", "headache", "head injury", "trauma", "chot", "gira", "khoon", "gir gaya", "takkar",
})


def detect_symptom_emergency(
    symptoms: str,
    gemini_client,
    groq_client,
    user_profile: dict,
    confirmed_past: bool = False,
) -> "EmDecision":
    """
    Detect emergency status for symptoms BEFORE generating or streaming guidance.
    - Pure classification via emergency.py
    - Devanagari LLM classifier check if rule engine finds tier 0
    - Strong past-tense check: sentinel tier -1 if not confirmed_past
    - confirmed_past=True: bypasses past-tense sentinel and downgrades tier 1 to 0
    - Tier-2 non-crisis/non-snakebite: second-layer LLM confirmation
    """
    from emergency import Decision
    dec = _em_classify(symptoms)

    # Devanagari recall boost: if Devanagari is present and rule returned 0, use LLM
    if is_devanagari(symptoms) and not dec.is_emergency:
        if llm_emergency_confirm(symptoms, gemini_client, groq_client):
            dec = Decision(tier=2, matched="llm-classifier (Devanagari)")

    # Past-event confirmation: user confirmed it was in the past
    if confirmed_past:
        if dec.tier == 1:
            dec = Decision(tier=0, matched=f"{dec.matched} (past event)")

    # Strong past-tense check for tier-1
    if dec.tier == 1 and not confirmed_past and is_strong_past_tense(symptoms):
        return Decision(tier=-1, matched=dec.matched)

    # Tier-2 LLM confirm: veto non-crisis, non-snakebite if LLM says clean NO
    if dec.tier == 2 and not dec.snakebite and not dec.crisis:
        confirmed = llm_emergency_confirm(symptoms, gemini_client, groq_client)
        if not confirmed:
            dec = Decision(tier=0, matched=dec.matched)

    return dec


def generate_symptom_response(
    symptoms: str,
    gemini_client,
    groq_client,
    user_profile: dict,
    chat_history: list[dict],
    dec: "EmDecision",
    stream: bool = True,
    stream_placeholder=None,
) -> str:
    """
    Generate triage response based on Decision.
    If snakebite, crisis, or hard tier-1 (non-info), returns canned guidance immediately.
    If info query, tier-2, or non-emergency, calls LLM (streamed into stream_placeholder).
    """
    region = user_profile.get("region", "India")
    if dec.snakebite:
        return SNAKEBITE_FIRST_AID

    if dec.crisis:
        return (
            "💙 **Please know you are not alone.**\n\n"
            f"{_crisis_copy(region)}\n\n"
            "AuraVoice is here with you. Please reach out to a counsellor right now."
        )

    if dec.tier == 1 and not dec.info_query:
        return (
            "🚨 **This sounds like a medical emergency.**\n\n"
            "Call 📞 [**112**](tel:112) or 🚑 [**108**](tel:108) immediately. "
            "Do not wait for AI guidance.\n\n"
            "⚠️ This is not a medical diagnosis. Please consult a qualified doctor."
        )

    # For information queries, tier-2, or non-emergency: call LLM!
    ctx = build_context(
        user_profile,
        ["age", "language", "conditions", "medicines", "allergies"]
    )
    messages_for_model = []
    for msg in chat_history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages_for_model.append(
                f"{'User' if msg['role'] == 'user' else 'AuraVoice'}: {msg['content'][:400]}"
            )
    history_block = "\n".join(messages_for_model)

    age_str = user_profile.get("age", "")
    try:
        age_num: Optional[int] = int(age_str) if age_str and age_str.strip().isdigit() else None
    except (ValueError, AttributeError):
        age_num = None

    meds = user_profile.get("medicines", [])
    anticoag = any(any(root in m.lower() for root in _ANTICOAG_ROOTS) for m in meds)
    symp_lower = symptoms.lower()
    anticoag_relevant = anticoag and any(w in symp_lower for w in _ANTICOAG_CONTEXT_WORDS)

    high_risk = (
        (age_num is not None and (age_num > 65 or age_num < 5))
        or anticoag_relevant
        or "pregnancy" in " ".join(user_profile.get("conditions", [])).lower()
    )
    urgency_note = (
        "⚠️ IMPORTANT: This patient is in a higher-risk group "
        "(age, anticoagulant use with bleeding/injury context, or pregnancy). "
        "Bias urgency level HIGHER. When in doubt, use 🔴 Urgent.\n\n"
    ) if high_risk else ""

    info_note = (
        "NOTE: The user is asking an informational / educational health question. "
        "Answer their question clearly and informatively while maintaining appropriate health safety guidance.\n\n"
    ) if dec.info_query else ""

    prompt = (
        f"{urgency_note}"
        f"{info_note}"
        f"Conversation so far:\n{history_block}\n\n"
        f"Current symptom report: \"{symptoms}\"\n\n"
        "Please provide:\n"
        "1. **Urgency** — 🔴 Urgent (go now) / 🟡 Soon (within 3 days) / 🟢 Routine\n"
        "2. **Possible causes** (2–3, non-diagnostic)\n"
        "3. **Self-care steps** (what to do right now at home)\n"
        "4. **Warning signs** that mean call 112 immediately\n"
        "5. **3 questions to ask the doctor**\n"
        "6. End with: '⚠️ This is not a medical diagnosis. Please consult a qualified doctor.'\n\n"
        "Plain language. Class 8 literacy. Hindi/regional if user writes in Hindi/regional."
    )
    return run_agent(
        prompt, None, gemini_client, groq_client, ctx,
        stream=stream, stream_placeholder=stream_placeholder
    )


def process_symptom_text(
    symptoms: str,
    gemini_client,
    groq_client,
    user_profile: dict,
    chat_history: list[dict],
    stream_placeholder=None,
    confirmed_past: bool = False,
) -> tuple["EmDecision", str]:
    """
    Text symptoms → tiered emergency check → structured triage.
    Returns (Decision, response_text).
    """
    dec = detect_symptom_emergency(
        symptoms, gemini_client, groq_client, user_profile, confirmed_past=confirmed_past
    )
    if dec.tier == -1:
        canned = (
            "🚨 **You mentioned something that can be serious.** "
            "Please confirm: **Is this happening right now?**\n\n"
            "If YES — call 📞 [**112**](tel:112) / 🚑 [**108**](tel:108) immediately.\n\n"
            "⚠️ This is not a medical diagnosis. Please consult a qualified doctor."
        )
        return dec, canned

    response = generate_symptom_response(
        symptoms, gemini_client, groq_client, user_profile, chat_history, dec,
        stream=(stream_placeholder is not None), stream_placeholder=stream_placeholder
    )
    return dec, response


# ─────────────────────────────────────────────────────────────────────────────
# 8. DOCTOR HANDOFF — verbatim patient words, no invented content
# ─────────────────────────────────────────────────────────────────────────────

def generate_handoff_summary(
    symptoms: str,
    duration: str,
    medicines: str,
    allergies: str,
    conditions: str,
    gemini_client,
    groq_client,
    user_profile: dict,
) -> str:
    """
    SOAP-lite handoff using ONLY what the patient reported.
    No invented questions — questions come from the patient's own words.
    """
    ctx = build_context(user_profile, ["age", "language", "region"])
    prompt = (
        "Generate a concise doctor handoff summary using ONLY the information below.\n"
        "Do NOT invent content, diagnoses, or questions not implied by what the patient said.\n\n"
        f"Patient's own words (symptoms): {symptoms}\n"
        f"Duration: {duration or 'not stated'}\n"
        f"Current medicines: {medicines or 'none reported'}\n"
        f"Allergies: {allergies or 'none reported'}\n"
        f"Known conditions: {conditions or 'none reported'}\n\n"
        "Format:\n"
        "## Patient Summary (for Doctor)\n"
        "**Patient says:** [verbatim or close paraphrase of their words]\n"
        "**Duration:** ...\n"
        "**Current Medications:** [list]\n"
        "**Known Allergies:** [list]\n"
        "**Known Conditions:** [list]\n"
        "**Patient's questions for you:** [derived only from what they said — if none, write 'Patient has not stated specific questions']\n\n"
        "Keep to one printed page. Do not add diagnoses, assessments, or recommendations."
    )
    return run_agent(prompt, None, gemini_client, groq_client, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 9. MEDICATION RECONCILIATION — structured JSON output
# ─────────────────────────────────────────────────────────────────────────────

RECONCILE_JSON_SCHEMA = {
    "medicines": [
        {
            "name": "string — verbatim from input",
            "dose": "string",
            "timing": "string — e.g. twice daily",
            "schedule": {"morning": "bool", "afternoon": "bool", "evening": "bool", "night": "bool"},
            "warnings": ["string"],
            "beers_flag": "bool — risky for >65",
            "interaction_flags": ["string — e.g. 'interacts with warfarin'"],
            "duplicate_of": "string or null",
            "jan_aushadhi_note": "string or null",
        }
    ],
    "unclear_timings": ["string — e.g. 'Medicine X timing ambiguous: SOS vs BD'"],
    "disclaimer": "string"
}


def parse_indian_prescription_shorthand(text: str) -> list[dict]:
    """
    Python regex parser for Indian prescription shorthand:
    - 1-0-1, 1-1-1, 1-0-0, 0-0-1, 0-1-0, 1-1-0, 0-1-1, 1-1-1-1
    - OD (once daily), BD/BID (twice daily), TDS/TID (thrice daily), QDS/QID (4 times)
    - HS (bedtime), SOS/PRN (as needed)
    - AC (before meals), PC (after meals)
    Ticks schedule boxes only when explicitly stated.
    """
    extracted = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        if line.startswith("#") or line.startswith("|") or line.lower().startswith("rx"):
            continue
        dose_match = re.search(r'\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|tablets?|caps?))\b', line, re.IGNORECASE)
        dose = dose_match.group(1) if dose_match else ""
        name_part = line[:dose_match.start()].strip() if dose_match else " ".join(line.split()[:2])
        name = re.sub(r'^[0-9]+[.)\s-]+', '', name_part).strip()
        if not name or len(name) < 2:
            continue

        num_match = re.search(r'\b([0-2])\s*[-–/]\s*([0-2])\s*[-–/]\s*([0-2])(?:\s*[-–/]\s*([0-2]))?\b', line)
        tokens_upper = re.findall(r'\b[A-Za-z]+\b', line.upper())

        morning = False
        afternoon = False
        evening = False
        night = False
        timing_parts = []

        if num_match:
            m = int(num_match.group(1))
            a = int(num_match.group(2))
            n = int(num_match.group(3))
            extra = int(num_match.group(4)) if num_match.group(4) else None
            morning, afternoon = (m > 0), (a > 0)
            if extra is not None:
                evening, night = (n > 0), (extra > 0)
            else:
                night = (n > 0)
            timing_parts.append(f"{m}-{a}-{n}")
        elif "TDS" in tokens_upper or "TID" in tokens_upper:
            morning, afternoon, night = True, True, True
            timing_parts.append("thrice daily (TDS)")
        elif "BD" in tokens_upper or "BID" in tokens_upper:
            morning, night = True, True
            timing_parts.append("twice daily (BD)")
        elif "QDS" in tokens_upper or "QID" in tokens_upper:
            morning, afternoon, evening, night = True, True, True, True
            timing_parts.append("four times daily (QDS)")
        elif "OD" in tokens_upper or "QD" in tokens_upper:
            if "HS" in tokens_upper or "NIGHT" in tokens_upper or "BEDTIME" in tokens_upper:
                night = True
                timing_parts.append("once daily at night (OD HS)")
            else:
                morning = True
                timing_parts.append("once daily (OD)")
        elif "HS" in tokens_upper or "BEDTIME" in tokens_upper:
            night = True
            timing_parts.append("at bedtime (HS)")
        elif "SOS" in tokens_upper or "PRN" in tokens_upper:
            timing_parts.append("as needed (SOS)")

        if "AC" in tokens_upper:
            timing_parts.append("before food (AC)")
        elif "PC" in tokens_upper:
            timing_parts.append("after food (PC)")

        timing_str = ", ".join(timing_parts) if timing_parts else "timing not specified"
        extracted.append({
            "name": name,
            "dose": dose,
            "timing": timing_str,
            "schedule": {
                "morning": morning,
                "afternoon": afternoon,
                "evening": evening,
                "night": night,
            },
            "warnings": [],
            "beers_flag": False,
            "interaction_flags": [],
            "duplicate_of": None,
            "jan_aushadhi_note": None,
        })
    return extracted


def reconcile_medications_text(
    medication_text: str,
    gemini_client,
    groq_client,
    user_profile: dict,
) -> str:
    """
    Paste discharge text → structured JSON → rendered Python table.
    Temperature 0 for consistency.
    """
    ctx = build_context(user_profile, ["age", "language", "conditions", "allergies"])
    schema_str = json.dumps(RECONCILE_JSON_SCHEMA, indent=2)

    # Separate system instruction for JSON-only output — avoids SYSTEM_PROMPT
    # asking for disclaimers/teach-back which break JSON.loads()
    recon_system = (
        "You are a structured-data extraction assistant. "
        "Output ONLY valid JSON. No prose, no markdown fences, no explanation. "
        "If a field is unknown, use null. "
        "The response must be parseable by Python json.loads()."
    )

    prompt = (
        "Parse the following medication list or discharge instructions into structured JSON.\n"
        "Return ONLY valid JSON matching this schema exactly:\n\n"
        f"{schema_str}\n\n"
        "Rules:\n"
        "- name: copy VERBATIM from input — do not auto-correct drug names.\n"
        "- Mark EVERY medicine from the input — do not skip any.\n"
        "- For ambiguous timings (SOS, HS, PRN, BD/TDS), add to unclear_timings.\n"
        "- beers_flag: true if the drug is on common Beers Criteria lists.\n"
        "- jan_aushadhi_note: mention the generic salt and that same-salt generics exist, "
        "  do NOT name specific brands. Label as AI-generated — user must ask pharmacist.\n"
        "- disclaimer must be exactly: "
        "'AI-generated summary. Confirm with your pharmacist or doctor.'\n\n"
        "Medication input:\n"
        f"{medication_text}"
    )

    # Use temperature=0 + JSON-only system instruction (no SYSTEM_PROMPT)
    raw = None
    if gemini_client and GEMINI_AVAILABLE:
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[inject_context(prompt, ctx)],
                config=genai_types.GenerateContentConfig(
                    system_instruction=recon_system,
                    temperature=0.0,
                    max_output_tokens=2048,
                ),
            )
            raw = resp.text
        except Exception as exc:
            log.error("Reconciliation Gemini failed: %s", exc)

    if not raw and groq_client and GROQ_AVAILABLE:
        raw = call_groq(
            groq_client,
            inject_context(prompt, ctx),
            system=(
                "You are a structured-data extraction assistant. "
                "Output ONLY valid JSON. No prose, no markdown fences, no explanation. "
                "The response must be parseable by Python json.loads()."
            ),
        )



    # Try to parse JSON and render as human-readable table
    data = None
    if raw:
        try:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            json_str = json_match.group(1) if json_match else raw.strip()
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "medicines" in parsed:
                data = parsed
        except Exception as e:
            log.warning("JSON parse failed in reconciliation: %s", e)

    # Fallback / augment with Python shorthand parser
    py_meds = parse_indian_prescription_shorthand(medication_text)
    if data is None:
        data = {
            "medicines": py_meds,
            "unclear_timings": [],
            "disclaimer": "AI-generated summary. Confirm with your pharmacist or doctor."
        }
    else:
        # Cross-check: Ensure EVERY input medicine appears in output
        existing_names = [
            str(m.get("name", "")).lower() for m in data.get("medicines", []) if isinstance(m, dict)
        ]
        for pm in py_meds:
            p_name = pm["name"].lower()
            matched_existing = None
            for em in data.get("medicines", []):
                if isinstance(em, dict) and p_name in str(em.get("name", "")).lower():
                    matched_existing = em
                    break
            if matched_existing:
                # If model had empty/false schedule but Python parser found explicit shorthand (e.g. 1-0-1),
                # tick schedule boxes only when stated
                sched_existing = matched_existing.get("schedule") or {}
                if not any(sched_existing.values()) and any(pm["schedule"].values()):
                    matched_existing["schedule"] = pm["schedule"]
                if pm["timing"] and (not matched_existing.get("timing") or matched_existing.get("timing") == "null"):
                    matched_existing["timing"] = pm["timing"]
                if pm["dose"] and not matched_existing.get("dose"):
                    matched_existing["dose"] = pm["dose"]
            else:
                data.setdefault("medicines", []).append(pm)

    try:
        report = _render_reconciliation(data)
        return report, data
    except Exception as exc:
        log.error("Render reconciliation failed: %s", exc)
        return raw or SAFE_FALLBACK_RESPONSE, data


def _spoken_reconciliation(data: Optional[dict]) -> str:
    """
    Build a TTS-friendly spoken summary from reconciliation JSON.
    The table is unreadable by TTS; we build prose directly from parsed data.
    """
    if not isinstance(data, dict):
        return ""
    parts = []
    for m in data.get("medicines", []) or []:
        if not isinstance(m, dict):
            continue
        name = m.get("name") or "Medicine"
        dose = m.get("dose") or ""
        sched = (m.get("schedule") if isinstance(m.get("schedule"), dict) else None) or {}
        times = []
        if sched.get("morning"):   times.append("morning")
        if sched.get("afternoon"): times.append("afternoon")
        if sched.get("evening"):   times.append("evening")
        if sched.get("night"):     times.append("night")
        time_str = " and ".join(times) if times else (m.get("timing") or "timing unclear")
        parts.append(f"{name} {dose}, take in the {time_str}.")
    return "Your medicines: " + " ".join(parts) if parts else ""


def _render_reconciliation(data: dict) -> str:
    """Render parsed reconciliation JSON as a Markdown report.
    Coerces null schedule/warnings to empty dict/list to prevent AttributeError."""
    if not isinstance(data, dict):
        return "⚠️ Could not parse medication schedule."

    lines = ["## 💊 Medication Schedule\n",
             "_⚠️ AI-generated. Confirm with your pharmacist or doctor._\n"]

    lines.append("\n| Medicine | Dose | Morning | Afternoon | Evening | Night | Warnings |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|---|")

    for m in data.get("medicines", []) or []:
        if not isinstance(m, dict):
            continue
        sched = (m.get("schedule") if isinstance(m.get("schedule"), dict) else None) or {}
        flags = []
        if m.get("beers_flag"):
            flags.append("⚠️ Beers (risk >65)")
        int_flags = m.get("interaction_flags")
        if isinstance(int_flags, list):
            flags.extend(int_flags)
        if m.get("duplicate_of"):
            flags.append(f"⚠️ Possible duplicate of {m['duplicate_of']}")
        warn_str = "; ".join(str(f) for f in flags) if flags else "—"
        lines.append(
            f"| **{m.get('name','?')}** | {m.get('dose','?')} | "
            f"{'✅' if sched.get('morning') else '—'} | "
            f"{'✅' if sched.get('afternoon') else '—'} | "
            f"{'✅' if sched.get('evening') else '—'} | "
            f"{'✅' if sched.get('night') else '—'} | "
            f"{warn_str} |"
        )

    unclear = data.get("unclear_timings") or []
    if isinstance(unclear, list) and unclear:
        lines.append("\n**⚠️ Unclear timings — ask your pharmacist:**")
        for u in unclear:
            lines.append(f"- {u}")

    jan = [m for m in (data.get("medicines") or []) if isinstance(m, dict) and m.get("jan_aushadhi_note")]
    if jan:
        lines.append("\n**💰 Affordable generic options (verify with pharmacist):**")
        for m in jan:
            lines.append(f"- **{m.get('name','')}**: {m.get('jan_aushadhi_note','')}")

    disc = data.get("disclaimer") or "AI-generated summary. Confirm with your pharmacist or doctor."
    lines.append(f"\n_{disc}_")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "session_id":       f"av_{int(time.time())}",
        "active_model":     "none",
        "last_error":       "",
        "chat_history":     [],
        "simple_mode":      False,
        "consent_given":    False,
        "self_test_done":   False,
        "self_test_result": {},
        "user_profile": {
            "name":       "User",   # stored locally only — never sent to LLM
            "age":        "",
            "region":     "India",
            "language":   "English",
            "conditions": [],
            "medicines":  [],
            "allergies":  [],
        },
        "api_keys_set":     False,
        "gemini_api_key":   "",
        "groq_api_key":     "",
        # Vision cache keyed by (file_hash, task_key) → (result_str, quality_dict)
        "vision_cache":     {},
        # Audio results
        "audio_transcript": None,
        "audio_response":   None,
        "audio_alert":      None,   # serialised Decision dict for rerun persistence
        # Doctor tools
        "handoff_result":   None,
        "recon_result":     None,
        "recon_spoken":     "",     # TTS-friendly prose from reconciliation JSON
        # Past-tense tier-1 confirmation pending
        "pending_tier1_check":        None,   # (matched_phrase, symptoms_text) or None
        "pending_audio_tier1_check":  None,   # (matched_phrase, audio_typed_text) or None
        "prefill_symptom_past":       None,   # confirmed-past symptoms pending processing
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# 11. TTS — clean markdown, safe JS escaping, correct locale codes
# ─────────────────────────────────────────────────────────────────────────────

_MD_CLEAN_RE = re.compile(
    r"\|[^\n]*"               # table rows
    r"|`{1,3}[^`]*`{1,3}"    # inline code / fences
    r"|#{1,6}\s"              # headings
    r"|\*{1,3}([^*]+)\*{1,3}"  # bold / italic (keep inner text)
    r"|_([^_]+)_"             # italic
    r"|!\[[^\]]*\]\([^)]*\)"  # images
    r"|\[[^\]]*\]\([^)]*\)"   # links
    r"|[*_~`>#\-]"            # misc symbols
)

# Unicode ranges for emoji removal — preserves Devanagari, Tamil, Telugu, Bengali, Arabic etc.
# Matches: Emoticons U+1F600–1F64F, Misc symbols U+1F300–1F5FF, Transport U+1F680–1F6FF,
# Supplemental U+1F900–1FAFF, Misc symbols block U+2600–26FF, Dingbats U+2700–27BF
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F900-\U0001FAFF"   # supplemental symbols
    "\U00002600-\U000026FF"   # misc symbols
    "\U00002700-\U000027BF"   # dingbats
    "\U0001F1E0-\U0001F1FF"   # flags
    "]",
    flags=re.UNICODE,
)

def _clean_for_tts(text: str) -> str:
    """
    Strip Markdown syntax and emoji for TTS.
    DOES NOT truncate — speak_text() chunks by sentence instead.
    Preserves Devanagari/Tamil/Telugu/Bengali (emoji regex only, NOT ascii encode).
    """
    # Remove think blocks
    text = _strip_think_blocks(text)
    # Remove table rows (lines starting with |) — they are unreadable aloud
    text = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("|"))
    # Remove markdown symbols but keep inner words
    text = _MD_CLEAN_RE.sub(lambda m: m.group(1) or m.group(2) or " ", text)
    # Remove ONLY emoji — preserve Unicode scripts (Devanagari, Tamil, etc.)
    text = _EMOJI_RE.sub(" ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text   # no truncation here; speak_text chunks instead


def speak_text(text: str, lang_code: str = "en-IN"):
    """
    Inject Web Speech API utterance via st.components.
    Uses json.dumps for safe JS string escaping.
    Chunks long text by sentence to work around browser 200-word limits.
    Works in Chrome/Edge/Safari desktop. Mobile iframe TTS may be blocked.
    """
    cleaned = _clean_for_tts(text)
    # Split into sentences for reliable playback (some browsers drop after ~200 words)
    sentences = re.split(r"(?<=[.!?।])\s+", cleaned)
    # Build JS array of utterances
    utterances_js = ", ".join(json.dumps(s) for s in sentences if s.strip())
    html = f"""
    <button id="av_tts_play" onclick="avSpeak()" style="
        padding:8px 18px;font-size:14px;border-radius:6px;
        background:#1a73e8;color:white;border:none;cursor:pointer;margin:2px;">
      ▶ Play
    </button>
    <button onclick="window.speechSynthesis && window.speechSynthesis.cancel()" style="
        padding:8px 18px;font-size:14px;border-radius:6px;
        background:#555;color:white;border:none;cursor:pointer;margin:2px;">
      ⏹ Stop
    </button>
    <sc` + `ript>
    function avSpeak() {{
      if (!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      var parts = [{utterances_js}];
      var idx = 0;
      function next() {{
        if (idx >= parts.length) return;
        var u = new SpeechSynthesisUtterance(parts[idx++]);
        u.lang = "{lang_code}";
        u.rate = 0.85;
        u.onend = next;
        window.speechSynthesis.speak(u);
      }}
      next();
    }}
    </sc` + `ript>
    """
    st.components.v1.html(html.replace("` + `", ""), height=60)


def tts_controls(text: str, key_prefix: str = "tts"):
    """
    Render inline Play/Stop controls using the HTML component (speak_text).
    The buttons live inside the HTML iframe — they work on user-gesture in all browsers.
    """
    user_lang = st.session_state["user_profile"].get("language", "English")
    lang_code = TTS_LANG_CODES.get(user_lang, "en-IN")
    speak_text(text, lang_code)


# ─────────────────────────────────────────────────────────────────────────────
# 12. CONSENT GATE
# ─────────────────────────────────────────────────────────────────────────────

def render_consent_gate() -> bool:
    if st.session_state.get("consent_given"):
        return True
    st.markdown(
        """
        <div style="border:2px solid #e5e7eb;border-radius:10px;padding:1.5rem;margin-bottom:1rem;">
        <h3 style="margin-top:0;">📋 Before You Begin · शुरू करने से पहले</h3>
        <p><strong>AuraVoice is a health information tool, not a medical service. · ऑरा-वॉइस एक स्वास्थ्य सूचना साधन है, चिकित्सीय सेवा नहीं।</strong></p>
        <hr style="margin:0.8rem 0;border:0;border-top:1px solid #e5e7eb;"/>
        <div style="display:flex;flex-wrap:wrap;gap:1.5rem;">
          <div style="flex:1;min-width:280px;">
            <h4 style="margin:0 0 0.5rem 0;">🇬🇧 English Notice</h4>
            <p style="font-size:0.92rem;">It does not diagnose, prescribe, or replace a doctor's advice.</p>
            <ul style="font-size:0.9rem;">
              <li>Your text, images and audio are processed by <strong>Google Gemini</strong> and <strong>Groq APIs</strong>. Do not enter real sensitive health data during demo.</li>
              <li>AuraVoice does not store data beyond this browser session.</li>
              <li>Designed in alignment with <strong>India's DPDP Act 2023</strong> (purpose limitation, informed consent).</li>
              <li>In a medical emergency, call <strong>112</strong> / <strong>108</strong> immediately.</li>
            </ul>
            <p style="font-size:0.85rem;color:#6b7280;"><em>Consent must be specific, in a language you understand, and can be withdrawn by closing this tab.</em></p>
          </div>
          <div style="flex:1;min-width:280px;border-left:1px solid #e5e7eb;padding-left:1rem;">
            <h4 style="margin:0 0 0.5rem 0;">🇮🇳 हिंदी सूचना (Hindi Notice)</h4>
            <p style="font-size:0.92rem;">यह ऐप किसी बीमारी का निदान नहीं करता और डॉक्टर की सलाह का विकल्प नहीं है।</p>
            <ul style="font-size:0.9rem;">
              <li>आपका टेक्स्ट, आवाज़ और तस्वीरें <strong>Google Gemini</strong> और <strong>Groq APIs</strong> द्वारा प्रोसेस की जाती हैं। डेमो में असली निजी स्वास्थ्य जानकारी न डालें।</li>
              <li>ऑरा-वॉइस इस ब्राउज़र सत्र के बाद कोई डेटा सुरक्षित नहीं रखता।</li>
              <li>यह <strong>भारत के DPDP अधिनियम 2023</strong> के सिद्धांतों (उद्देश्य सीमा, सूचित सहमति) के अनुरूप है।</li>
              <li>चिकित्सा आपातकाल में तुरंत <strong>112</strong> या <strong>108</strong> पर कॉल करें।</li>
            </ul>
            <p style="font-size:0.85rem;color:#6b7280;"><em>सहमति विशिष्ट है, आपकी समझी जाने वाली भाषा में है, और इस टैब को बंद करके वापस ली जा सकती है।</em></p>
          </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("✅ I understand & Consent · मैं समझता हूँ और सहमत हूँ", use_container_width=True, type="primary"):
        st.session_state["consent_given"] = True
        st.rerun()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 13. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def _get_secrets() -> tuple[str, str]:
    """Safely read st.secrets without raising if no secrets file exists."""
    try:
        gk  = st.secrets.get("GEMINI_API_KEY", "") or ""
        grk = st.secrets.get("GROQ_API_KEY",   "") or ""
        return gk, grk
    except Exception:
        return "", ""


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stethoscope.png", width=52)
        st.title("AuraVoice ⚕️")
        st.caption("Multimodal Health & Accessibility Agent")

        st.session_state["simple_mode"] = st.toggle(
            "🔤 Simple Mode (large text)",
            value=st.session_state.get("simple_mode", False),
            help="Larger text, bigger buttons — for elderly or low-vision users",
        )
        st.divider()

        # ── API Keys (hidden when in production via AURAVOICE_PROD env flag) ──
        is_prod = os.getenv("AURAVOICE_PROD", "").lower() in ("1", "true", "yes")
        if not is_prod:
            with st.expander("🔑 API Keys (Dev)", expanded=not st.session_state["api_keys_set"]):
                st.info(
                    "Keys load from `st.secrets` on Streamlit Cloud. "
                    "Enter here for local dev only.",
                    icon="ℹ️",
                )
                gemini_input = st.text_input(
                    "Gemini API Key", type="password",
                    value=st.session_state.get("gemini_api_key", ""),
                )
                groq_input = st.text_input(
                    "Groq API Key", type="password",
                    value=st.session_state.get("groq_api_key", ""),
                )
                if st.button("🔗 Connect & Test APIs", use_container_width=True):
                    with st.spinner("Connecting…"):
                        init_gemini.clear()
                        init_groq.clear()
                        gk  = gemini_input.strip()
                        grk = groq_input.strip()
                        gm  = init_gemini(gk)  if gk  else None
                        gc  = init_groq(grk)   if grk else None
                        st.session_state["gemini_api_key"] = gk  if gm else ""
                        st.session_state["groq_api_key"]   = grk if gc else ""
                        st.session_state["api_keys_set"]   = gm is not None or gc is not None
                        if st.session_state["api_keys_set"]:
                            test = startup_self_test(gm, gc)
                            st.session_state["self_test_done"]   = True
                            st.session_state["self_test_result"] = test
                    if st.session_state["api_keys_set"]:
                        res = st.session_state["self_test_result"]
                        gemini_status = "✅" if res.get("gemini_ok") else "⚠️ (init OK, call failed)"
                        groq_status   = "✅" if res.get("groq_ok")   else "⚠️ (init OK, call failed)"
                        st.success(f"Connected! Gemini: {gemini_status}  Groq: {groq_status}")
                    else:
                        st.error("Connection failed. Check your keys.")

        st.divider()

        # ── User Profile ──────────────────────────────────────────────────────
        st.subheader("👤 Profile")
        p = st.session_state["user_profile"]
        name_v    = st.text_input("Name",  value=p["name"])
        age_v     = st.text_input("Age",   value=p["age"])

        region_opts = list(INDIA_HOTLINES.keys()) + ["Global", "USA", "UK", "Australia"]
        # Use simplified region list for matching emergency hotlines
        simple_regions = ["India", "Global", "USA", "UK", "Australia"]
        region_v  = st.selectbox(
            "Region",
            options=simple_regions,
            index=simple_regions.index(p["region"]) if p["region"] in simple_regions else 0,
        )
        lang_opts = ["English", "Hindi", "Tamil", "Telugu", "Bengali",
                     "Bhojpuri", "Awadhi", "Punjabi",
                     "Spanish", "French", "Arabic", "Mandarin", "Portuguese", "Swahili"]
        lang_v    = st.selectbox(
            "Language",
            options=lang_opts,
            index=lang_opts.index(p["language"]) if p["language"] in lang_opts else 0,
        )
        cond_v    = st.text_area("Known conditions",  height=52,
                                 value=", ".join(p.get("conditions", [])),
                                 placeholder="diabetes, hypertension…")
        med_v     = st.text_area("Current medicines", height=52,
                                 value=", ".join(p.get("medicines", [])),
                                 placeholder="Metformin 500mg, Amlodipine 5mg…")
        allergy_v = st.text_input("Allergies",
                                   value=", ".join(p.get("allergies", [])),
                                   placeholder="Penicillin, Sulfa…")

        if st.button("💾 Save Profile", use_container_width=True):
            p["name"]       = name_v
            p["age"]        = age_v
            p["region"]     = region_v
            p["language"]   = lang_v
            p["conditions"] = [c.strip() for c in cond_v.split(",")     if c.strip()]
            p["medicines"]  = [m.strip() for m in med_v.split(",")      if m.strip()]
            p["allergies"]  = [a.strip() for a in allergy_v.split(",")  if a.strip()]
            st.success(f"✅ Saved for {name_v}!")

        st.divider()

        # ── Model Status ──────────────────────────────────────────────────────
        st.subheader("⚙️ Status")
        active = st.session_state.get("active_model", "none")
        if GEMINI_MODEL in active:
            st.success(f"Active: {active}", icon="🟢")
        elif "groq" in active:
            st.warning(f"Active: {active}", icon="🟡")
        elif "fallback" in active:
            st.error("Static fallback mode", icon="🔴")
            last_err = st.session_state.get("last_error", "")
            if last_err:
                st.caption(f"ℹ️ {last_err}")
        else:
            st.info("No query sent yet", icon="⚪")

        st.divider()
        st.caption(
            "⚠️ AuraVoice is a health information tool — not a medical service. "
            "Always consult a doctor for medical decisions."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 14. SIMPLE MODE CSS
# ─────────────────────────────────────────────────────────────────────────────

def inject_simple_mode_css():
    st.markdown("""
    <style>
    body, .stMarkdown, p, li, td, th { font-size:1.35rem !important; line-height:1.9 !important; }
    .stButton > button { font-size:1.3rem !important; padding:0.8rem 1.4rem !important;
        background:#1a73e8 !important; color:white !important; font-weight:bold !important; }
    .stChatMessage { font-size:1.3rem !important; }
    h1 { font-size:2.5rem !important; } h2 { font-size:2rem !important; }
    h3 { font-size:1.7rem !important; }
    </style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 15. INDIA PATHWAYS PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_india_pathways():
    with st.expander("🏥 India Health Services — Who to Call", expanded=False):
        rows = "\n".join(f"| {s} | **{n}** |" for s, n in INDIA_HOTLINES.items())
        st.markdown(
            f"| Service | Number |\n|---|---|\n{rows}\n\n"
            "_⚠️ Verify all numbers at [mohfw.gov.in](https://mohfw.gov.in) before relying on them. "
            "Last checked: Oct 2026._"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 16. UI TABS
# ─────────────────────────────────────────────────────────────────────────────

def render_hero():
    active = st.session_state.get("active_model", GEMINI_MODEL)
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
                padding:1.8rem 2.4rem;border-radius:12px;margin-bottom:1.2rem;color:white;">
            <h1 style="margin:0;font-size:2rem;">🎙️ AuraVoice</h1>
            <p style="margin:0.3rem 0 0;opacity:0.85;">
                Multimodal Health &amp; Accessibility Agent</p>
            <p style="margin:0.4rem 0 0;font-size:0.78rem;opacity:0.5;">
                {active} · Aura Voice </p>
        </div>""",
        unsafe_allow_html=True,
    )
    # Demo / shared-key warning
    st.info(
        "🔬 **Demo mode** — Do not enter real personal health data. "
        "Use the sample cases below or describe fictional scenarios.",
        icon="⚠️",
    )


def tab_symptom_checker(gemini_client, groq_client):
    st.subheader("💬 Symptom Checker")
    st.caption(
        "Describe symptoms in any language. AuraVoice gives structured triage guidance. "
        "Not a diagnosis — always consult a doctor."
    )

    # Banner for languages where emergency detection coverage is incomplete
    user_lang = st.session_state["user_profile"].get("language", "English")
    if user_lang in _LANGUAGES_WITH_NO_EMERGENCY_COVERAGE:
        st.warning(
            f"⚠️ Emergency phrase detection is **not available in {user_lang}**. "
            "If you have a medical emergency, call your local emergency number directly.",
            icon="⚠️",
        )

    render_india_pathways()

    # Sample cases for demo
    with st.expander("📋 Sample cases (click to load)", expanded=False):
        samples = {
            "Fever (English)":           "I have had fever of 102°F since yesterday with body ache.",
            "Dizziness (Hindi)":         "mujhe do din se chakkar aa rahe hain aur kamzori lag rahi hai",
            "Chest tightness (English)": "I have chest tightness when climbing stairs, started today.",
            "Snake bite (Hindi)":        "mere haath par koi cheez ne kata, shayad saanp tha",
        }
        cols = st.columns(len(samples))
        for col, (label, text) in zip(cols, samples.items()):
            with col:
                if st.button(label, use_container_width=True, key=f"sample_{label}"):
                    st.session_state["prefill_symptom"] = text

    # Display chat history — re-render emergency banners stored as "alert" role messages
    for msg in st.session_state["chat_history"]:
        role = msg.get("role", "user")
        if role == "alert":
            # Reconstruct Decision from stored fields for render_emergency_alert
            from emergency import Decision
            stored_dec = Decision(
                tier=msg.get("tier", 2),
                matched=msg.get("matched", ""),
                crisis=msg.get("crisis", False),
                snakebite=msg.get("snakebite", False),
                info_query=msg.get("info_query", False),
            )
            render_emergency_alert(
                stored_dec,
                st.session_state["user_profile"]["region"],
            )
        else:
            with st.chat_message(role):
                st.markdown(msg["content"])
                if role == "assistant":
                    tts_controls(msg["content"], key_prefix=f"tts_{msg.get('id','x')}")

    # Handle pending tier-1 past-tense confirmation widget
    pending = st.session_state.get("pending_tier1_check")
    if pending:
        matched_p, symptoms_p = pending
        # Show the tier-1 alert FIRST (always visible), then the confirmation widget
        from emergency import Decision
        render_emergency_alert(
            Decision(tier=1, matched=matched_p),
            st.session_state["user_profile"]["region"],
        )
        st.warning(
            f"This may be a past event. Please confirm: **Is this happening RIGHT NOW?**",
            icon="🚨",
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ YES — happening NOW", use_container_width=True,
                         type="primary", key="tier1_yes"):
                st.session_state["pending_tier1_check"] = None
                st.session_state["chat_history"].append({
                    "role": "alert", "tier": 1, "matched": matched_p,
                    "crisis": False, "snakebite": False,
                    "id": str(int(time.time() * 1000))
                })
                st.rerun()
        with col_no:
            if st.button("❌ NO — it was in the past", use_container_width=True,
                         key="tier1_no"):
                st.session_state["pending_tier1_check"] = None
                # Re-process with confirmed_past=True — no loop possible
                st.session_state["prefill_symptom_past"] = symptoms_p
                st.rerun()

    # Process "confirmed past" submission (tier-1 that user confirmed is past)
    past_input = st.session_state.pop("prefill_symptom_past", None)

    prefill = st.session_state.pop("prefill_symptom", "")
    user_input = st.chat_input(
        "Describe your symptoms in any language…",
        key="symptom_input",
    )
    # Allow prefill to act as input
    if prefill and not user_input:
        user_input = prefill

    # If past-event confirmed, process it normally (confirmed_past=True)
    if past_input and not user_input:
        user_input = past_input
        confirmed_past_flag = True
    else:
        confirmed_past_flag = False

    if user_input:
        msg_id = str(int(time.time() * 1000))
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input, "id": msg_id}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # 1. Split detection out before assistant bubble & LLM streaming
        region = st.session_state["user_profile"]["region"]
        dec = detect_symptom_emergency(
            user_input,
            gemini_client, groq_client,
            st.session_state["user_profile"],
            confirmed_past=confirmed_past_flag,
        )

        if dec.tier == -1:
            # Past-tense sentinel: queue the confirmation widget and rerun immediately
            st.session_state["pending_tier1_check"] = (dec.matched, user_input)
            st.rerun()

        # 2. Render emergency alert FIRST (above the assistant bubble, before any LLM call)
        if dec.is_emergency:
            st.session_state["chat_history"].append({
                "role": "alert",
                "tier": max(dec.tier, 2),
                "matched": dec.matched,
                "crisis": dec.crisis,
                "snakebite": dec.snakebite,
                "info_query": dec.info_query,
                "id": msg_id + "_alert",
            })
            render_emergency_alert(dec, region)

        # 3. Open assistant chat bubble and generate / stream response
        with st.chat_message("assistant"):
            stream_ph = st.empty()  # placeholder Gemini streams into inside the bubble

            response = generate_symptom_response(
                user_input,
                gemini_client, groq_client,
                st.session_state["user_profile"],
                st.session_state["chat_history"][:-1],
                dec,
                stream=True,
                stream_placeholder=stream_ph,
            )

            # Always call stream_ph.markdown(response) at end (idempotent, ensures canned or Groq shows)
            stream_ph.markdown(response)

            st.caption(f"_Model: {st.session_state.get('active_model','?')}_")
            tts_controls(response, key_prefix=f"tts_{msg_id}_out")

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response, "id": msg_id + "_r"}
        )

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history"] = []
            st.session_state["pending_tier1_check"] = None
            st.rerun()


def tab_vision_agent(gemini_client, groq_client):
    st.subheader("📷 Vision Analysis")
    task_options = {
        "💊 Read Prescription / Label":          "prescription",
        "🏠 Describe Environment (Accessibility)": "environment",
        "🔍 Analyse Visible Symptom (Photo)":    "symptoms_photo",
        "📄 Read Medical Document / Report":     "document",
        "🏥 Discharge Summary / Med Sheet":      "discharge",
    }
    col1, col2 = st.columns([1, 1])
    with col1:
        task_label = st.selectbox("What to analyse?", list(task_options.keys()), key="vis_task")
        task_key   = task_options[task_label]
    with col2:
        if task_key == "environment":
            st.warning(
                "⚠️ Distance estimates from a single photo are approximate. "
                "Do not rely on them near stairs, traffic, or uneven ground. "
                "Consider Seeing AI or Be My Eyes for real-time navigation.",
                icon="⚠️",
            )
        elif task_key == "symptoms_photo":
            st.warning(
                "⚠️ Photo-based symptom assessment is unreliable and may perform "
                "poorly across skin tones. Always see a doctor.",
                icon="⚠️",
            )
        else:
            st.info("📸 Good lighting + steady hand = better results.", icon="💡")

    # Mobile: prefer file_uploader (opens native camera on mobile)
    input_method = st.radio(
        "Input method",
        ["🖼️ Upload / Camera (mobile-friendly)", "📷 Webcam (desktop)"],
        horizontal=True,
        key="vis_method",
    )
    image_file = (
        st.file_uploader(
            "Upload image or take photo",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="img_upload",
        )
        if "Upload" in input_method
        else st.camera_input("Take a photo", key="cam_capture")
    )

    # Analyse button — result keyed by (file_hash, task) to prevent stale results
    if image_file:
        file_bytes = image_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
        cache_key = (file_hash, task_key)
        if st.button("🔍 Analyse Image", use_container_width=True, type="primary", key="vis_analyse"):
            with st.spinner("Analysing…"):
                result, quality = process_camera_input(
                    image_file, task_key, gemini_client, groq_client,
                    st.session_state["user_profile"],
                )
            st.session_state["vision_cache"][cache_key] = (result, quality)

    # Render from cache — keyed by (file_hash, task) so new photo/task clears old result
    vision_cache = st.session_state.get("vision_cache", {})
    cached = vision_cache.get(cache_key) if image_file else None
    result  = cached[0] if cached else None
    quality = cached[1] if cached else {}
    if result and image_file:
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image_file, caption="Input image", use_container_width=True)
        with col_res:
            if quality:
                ok_str = "🟢 Good" if quality.get("ok") else "🟡 Low quality"
                st.caption(
                    f"Quality: {ok_str} · sharpness {quality.get('sharpness','?')} · "
                    f"brightness {quality.get('brightness','?')}"
                )
                if not quality.get("ok"):
                    st.warning(f"📸 {quality.get('guidance','')}", icon="📸")
            st.markdown("### 📋 Result")
            st.markdown(result)
            st.caption(f"_Model: {st.session_state.get('active_model','?')}_")
            tts_controls(result, key_prefix="tts_vision")
            st.download_button(
                "📥 Download Report", data=result,
                file_name=f"auravoice_{task_key}_report.txt",
                mime="text/plain", use_container_width=True,
            )


def tab_audio_agent(gemini_client, groq_client):
    st.subheader("🎙️ Voice Health Query")
    st.caption("Record or upload audio. AuraVoice transcribes and responds.")
    st.info(
        "**Speak clearly.** Mention what you feel, how long, and any medicines you take.",
        icon="🎤",
    )

    audio_bytes = None
    try:
        from streamlit_mic_recorder import mic_recorder  # type: ignore
        recording   = mic_recorder(
            start_prompt="⏺️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            key="mic_rec",
        )
        audio_bytes = recording["bytes"] if recording else None
    except Exception as e:
        log.warning("mic_recorder: %s", e)
        st.warning("In-browser mic unavailable. Upload audio or use the text box below.", icon="⚠️")

    uploaded = st.file_uploader(
        "Or upload audio (WAV is most reliable; MP3/OGG need Groq Whisper key)",
        type=["wav", "mp3", "ogg", "m4a", "webm"],
        key="audio_upload",
    )
    # Use getvalue() so we can re-read even after .read() on upload
    audio_source = audio_bytes or (uploaded.getvalue() if uploaded else None)

    if audio_source and len(audio_source) > 500:
        if st.button("🔊 Process Audio", use_container_width=True, type="primary", key="audio_process"):
            with st.spinner("Transcribing…"):
                transcript, response, dec = process_audio_input(
                    audio_source, gemini_client, groq_client,
                    st.session_state["user_profile"],
                )
            st.session_state["audio_transcript"] = transcript
            st.session_state["audio_response"]   = response
            # Store or clear decision for re-render
            if dec.is_emergency:
                st.session_state["audio_alert"] = {
                    "tier": dec.tier, "matched": dec.matched,
                    "crisis": dec.crisis, "snakebite": dec.snakebite,
                    "info_query": dec.info_query,
                }
            else:
                st.session_state["audio_alert"] = None
            st.rerun()

    # Handle pending past-event check in Voice tab if active
    pending_v = st.session_state.get("pending_audio_tier1_check")
    if pending_v:
        matched_pv, query_pv = pending_v
        from emergency import Decision
        render_emergency_alert(
            Decision(tier=1, matched=matched_pv),
            st.session_state["user_profile"]["region"],
        )
        st.warning(
            f"This may describe a past event. Please confirm: **Is this happening RIGHT NOW?**",
            icon="🚨",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ YES — happening NOW", use_container_width=True, type="primary", key="v_tier1_yes"):
                st.session_state["pending_audio_tier1_check"] = None
                st.session_state["audio_alert"] = {
                    "tier": 1, "matched": matched_pv, "crisis": False, "snakebite": False
                }
                st.rerun()
        with c2:
            if st.button("❌ NO — it was in the past", use_container_width=True, key="v_tier1_no"):
                st.session_state["pending_audio_tier1_check"] = None
                st.session_state["audio_alert"] = None
                # Reprocess as confirmed past
                resp_past = generate_symptom_response(
                    query_pv, gemini_client, groq_client,
                    st.session_state["user_profile"],
                    st.session_state["chat_history"],
                    Decision(tier=0, matched="confirmed past event"),
                    stream=False,
                )
                st.session_state["audio_response"] = resp_past
                st.rerun()

    # Render alert from cache so it persists on rerun
    audio_alert = st.session_state.get("audio_alert")
    if audio_alert:
        from emergency import Decision
        a_dec = Decision(
            tier=audio_alert.get("tier", 1),
            matched=audio_alert.get("matched", ""),
            crisis=audio_alert.get("crisis", False),
            snakebite=audio_alert.get("snakebite", False),
            info_query=audio_alert.get("info_query", False),
        )
        render_emergency_alert(a_dec, st.session_state["user_profile"]["region"])

    # Render from cache
    transcript = st.session_state.get("audio_transcript")
    response   = st.session_state.get("audio_response")
    if transcript:
        st.markdown("### 📝 Transcript / Query")
        st.info(transcript)
    if response:
        st.markdown("### 💡 AuraVoice Response")
        st.markdown(response)
        st.caption(f"_Model: {st.session_state.get('active_model','?')}_")
        tts_controls(response, key_prefix="tts_audio")

    # Text fallback
    st.divider()
    st.markdown("**No microphone? Type your query:**")
    typed = st.text_area(
        "Type what you would say…", height=90, key="audio_text_fb",
        placeholder="e.g., 'mujhe do din se bukhaar hai'",
    )
    if st.button("🔊 Process as Voice Query", use_container_width=True, key="typed_voice"):
        if typed.strip():
            dec_v = detect_symptom_emergency(
                typed, gemini_client, groq_client,
                st.session_state["user_profile"],
            )
            st.session_state["audio_transcript"] = typed

            if dec_v.tier == -1:
                st.session_state["pending_audio_tier1_check"] = (dec_v.matched, typed)
                st.session_state["audio_response"] = (
                    "🚨 Potential emergency reported. Please confirm above if this is happening right now."
                )
                st.rerun()

            if dec_v.is_emergency:
                st.session_state["audio_alert"] = {
                    "tier": max(dec_v.tier, 2),
                    "matched": dec_v.matched,
                    "crisis": dec_v.crisis,
                    "snakebite": dec_v.snakebite,
                    "info_query": dec_v.info_query,
                }
            else:
                st.session_state["audio_alert"] = None

            resp_v = generate_symptom_response(
                typed, gemini_client, groq_client,
                st.session_state["user_profile"],
                st.session_state["chat_history"],
                dec_v,
                stream=False,  # Single rendering — no double printing
            )
            st.session_state["audio_response"] = resp_v
            st.rerun()
        else:
            st.warning("Please type a query first.")


def tab_doctor_tools(gemini_client, groq_client):
    st.subheader("🏥 Doctor Tools")
    st.caption(
        "Use before a consultation or after discharge. "
        "All output is AI-generated — confirm with your pharmacist or doctor."
    )

    inner_t1, inner_t2 = st.tabs(["📋 Doctor Handoff Summary", "💊 Medication Reconciliation"])

    with inner_t1:
        st.markdown(
            "Fill in what the patient says. AuraVoice creates a one-page summary "
            "using **their own words** — no invented content."
        )
        p        = st.session_state["user_profile"]
        symptoms = st.text_area("Symptoms in patient's own words", height=80,
                                 placeholder="Fever for 3 days, headache, joint pain")
        duration = st.text_input("Duration", placeholder="3 days / 2 weeks")
        meds_val = st.text_area("Current medicines",
                                 value=", ".join(p.get("medicines", [])), height=52)
        allergy_val = st.text_input("Allergies", value=", ".join(p.get("allergies", [])))
        cond_val    = st.text_input("Known conditions", value=", ".join(p.get("conditions", [])))

        if st.button("📋 Generate Handoff", use_container_width=True, type="primary", key="gen_handoff"):
            if symptoms.strip():
                with st.spinner("Generating…"):
                    result = generate_handoff_summary(
                        symptoms, duration, meds_val, allergy_val, cond_val,
                        gemini_client, groq_client, p,
                    )
                st.session_state["handoff_result"] = result
            else:
                st.warning("Please enter symptoms.")

        # Render from cache
        handoff = st.session_state.get("handoff_result")
        if handoff:
            st.markdown(handoff)
            tts_controls(handoff, key_prefix="tts_handoff")
            st.download_button(
                "📥 Download Summary", data=handoff,
                file_name="auravoice_doctor_summary.txt", mime="text/plain",
                use_container_width=True,
            )

    with inner_t2:
        st.markdown(
            "Paste your discharge sheet or medicine list. "
            "AuraVoice creates a plain-language daily schedule and flags concerns.\n\n"
            "_⚠️ AI-generated. Drug names copied verbatim — always confirm with pharmacist._"
        )
        med_text = st.text_area(
            "Paste medicine list / discharge instructions",
            height=180,
            key="recon_input",
            placeholder=(
                "Metformin 500mg twice daily after meals\n"
                "Amlodipine 5mg once at night\n"
                "Aspirin 75mg once daily after breakfast\n"
                "..."
            ),
        )

        if st.button("💊 Reconcile Medications", use_container_width=True, type="primary", key="gen_recon"):
            if med_text.strip():
                with st.spinner("Analysing…"):
                    result, data = reconcile_medications_text(
                        med_text, gemini_client, groq_client,
                        st.session_state["user_profile"],
                    )
                st.session_state["recon_result"] = result
                # Store spoken text derived from JSON (not from the table)
                st.session_state["recon_spoken"] = (
                    _spoken_reconciliation(data) if data else ""
                )
            else:
                st.warning("Please paste your medicine list.")

        # Render from cache
        recon = st.session_state.get("recon_result")
        recon_spoken = st.session_state.get("recon_spoken", "")
        if recon:
            st.markdown(recon)
            # TTS uses the spoken prose summary, not the table
            tts_controls(recon_spoken or recon, key_prefix="tts_recon")
            st.download_button(
                "📥 Download Schedule", data=recon,
                file_name="auravoice_medication_schedule.txt", mime="text/plain",
                use_container_width=True,
            )


def tab_about():
    st.subheader("About AuraVoice")
    st.markdown(f"""
AuraVoice is a health information and accessibility tool for elderly, visually impaired,
and rural patients. It is an MVP built for people seeking immediate or any medical help.

### 🎯 Features
| Feature | Technology |
|---|---|
| Prescription reading (confidence + verbatim) | Gemini 3.6 Flash Vision |
| Tiered emergency detection (EN/HI/regional) | Rule engine, no false-suppress |
| Symptom triage with conversational memory | Gemini + Groq |
| Medication reconciliation (JSON + table) | Gemini at temperature 0 |
| Doctor handoff (patient's own words) | Gemini + Groq |
| In-browser mic + Whisper fallback | streamlit-mic-recorder + Groq Whisper |
| TTS (markdown-stripped, safe JS) | Web Speech API |
| Simple Mode | CSS injection |
| India pathways (112/108/102/Tele-MANAS) | Verified static list |
| Consent gate (DPDP 2023) | Session-state gate |

### ⚙️ Active Models
- Primary: `{GEMINI_MODEL}` (configurable via `AURAVOICE_GEMINI_MODEL` env)
- Fallback: `{GROQ_MODEL}` (configurable via `AURAVOICE_GROQ_MODEL` env)

### ⚠️ Limitations
- Emergency phrases cover EN/HI/Bhojpuri/Awadhi/Bengali/Tamil/Telugu. Not exhaustive.
- No negation suppression — "seene mein dard nahi" will trigger tier-2 (intentional — false negatives are more dangerous than false positives).
- Prescription reading is not verified against any drug database in this MVP.
- Has not been clinically evaluated or tested with real patients.
- TTS via Web Speech API may not work in mobile iframes — test on target device.
- Not a medical device. Not cleared by any regulatory authority.

### 🔒 Privacy
- No data stored beyond this browser session.
- Inputs sent to Google Gemini and Groq APIs (see their privacy policies).
- Designed to align with India's DPDP Act 2023.
- ABDM/ABHA: users encouraged to create digital health ID at abdm.gov.in.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 17. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AuraVoice — Health & Accessibility",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Auto-load from st.secrets (Streamlit Cloud)
    if not st.session_state["api_keys_set"]:
        gk, grk = _get_secrets()
        if gk:
            st.session_state["gemini_api_key"] = gk
        if grk:
            st.session_state["groq_api_key"] = grk
        if gk or grk:
            st.session_state["api_keys_set"] = True

    if st.session_state.get("simple_mode"):
        inject_simple_mode_css()

    if not render_consent_gate():
        return

    render_sidebar()
    render_hero()

    gk  = st.session_state.get("gemini_api_key", "")
    grk = st.session_state.get("groq_api_key", "")
    gemini_client = init_gemini(gk)  if gk  else None
    groq_client   = init_groq(grk)   if grk else None

    if not st.session_state["api_keys_set"]:
        st.warning(
            "👈 Enter API keys in the sidebar (expand 🔑 API Keys) and click **Connect**.",
            icon="🔑",
        )
        st.info(
            "Gemini (free): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n\n"
            "Groq (free): [console.groq.com/keys](https://console.groq.com/keys)",
            icon="🚀",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Symptoms",
        "📷 Vision",
        "🎙️ Voice",
        "🏥 Doctor Tools",
        "ℹ️ About",
    ])
    with tab1:
        tab_symptom_checker(gemini_client, groq_client)
    with tab2:
        tab_vision_agent(gemini_client, groq_client)
    with tab3:
        tab_audio_agent(gemini_client, groq_client)
    with tab4:
        tab_doctor_tools(gemini_client, groq_client)
    with tab5:
        tab_about()


if __name__ == "__main__":
    main()
