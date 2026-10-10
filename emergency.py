"""
emergency.py — AuraVoice emergency detection module
====================================================
Pure Python: no Streamlit imports, no LLM calls.
Importable by app.py, evaluation harness, and pytest.

Public API
----------
    classify(text: str) -> Decision

Decision fields:
    tier      : int   — 0=none, 1=CALL NOW, 2=URGENT
    matched   : str   — phrase or pattern that matched
    crisis    : bool  — self-harm / suicidal ideation (never veto)
    snakebite : bool  — envenomation (canned first-aid, no LLM)
    info_query: bool  — keyword inside an information question (show banner + still answer)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    tier:       int  = 0
    matched:    str  = ""
    crisis:     bool = False
    snakebite:  bool = False
    info_query: bool = False   # true = info question; show banner but still answer

    @property
    def is_emergency(self) -> bool:
        return self.tier > 0 or self.crisis or self.snakebite


# ---------------------------------------------------------------------------
# PHRASE LISTS
# Format: (phrase_lower, language_tag)
# ---------------------------------------------------------------------------

# Tier 1 = CALL NOW — unambiguous life-threat
TIER1_PHRASES: list[tuple[str, str]] = [
    # English – cardiac
    ("heart attack", "en"), ("cardiac arrest", "en"),
    ("crushing chest pain", "en"), ("chest pain radiating", "en"),
    ("pain in left arm", "en"), ("chest is being crushed", "en"),
    ("chest crushing", "en"),
    # English – respiratory arrest
    ("not breathing", "en"), ("stopped breathing", "en"),
    ("cannot breathe at all", "en"), ("airway blocked", "en"),
    ("choking", "en"),
    # English – stroke (FAST)
    ("face drooping", "en"), ("face is drooping", "en"),
    ("arm weakness suddenly", "en"), ("arm weakness", "en"),
    ("speech slurred", "en"), ("sudden vision loss", "en"),
    ("worst headache of my life", "en"),
    # English – severe bleeding / shock
    ("bleeding won't stop", "en"), ("bleeding profusely", "en"),
    ("unconscious", "en"), ("unresponsive", "en"), ("not waking up", "en"),
    ("no pulse", "en"), ("cannot feel his pulse", "en"), ("cannot feel a pulse", "en"),
    ("not responding", "en"),
    # English – poisoning / overdose
    ("overdose", "en"), ("swallowed poison", "en"), ("took too many pills", "en"),
    # English – seizure ongoing
    ("seizure won't stop", "en"), ("convulsions", "en"),
    # English – active suicidal crisis (tier-1 specific phrasing)
    ("going to kill myself", "en"), ("about to end my life", "en"),
    ("taking pills to die", "en"),
    # English – obstetric
    ("eclampsia", "en"), ("baby not moving for hours", "en"),
    ("heavy bleeding in pregnancy", "en"),
    # English – envenomation (snakebite / scorpion sting are ALWAYS Tier 1)
    ("bitten by snake", "en"), ("bitten by a snake", "en"),
    ("snakebite", "en"), ("snake bite", "en"),
    ("scorpion sting severe", "en"), ("scorpion sting", "en"),
    # Hindi – cardiac
    ("dil ka daura pad gaya", "hi"), ("dil ka dora", "hi"),
    ("seene mein bahut tez dard", "hi"), ("chhati mein tez dard", "hi"),
    ("seene mein dard aa raha hai", "hi"),
    ("दिल का दौरा", "hi"), ("सीने में बहुत तेज दर्द", "hi"),
    # Hindi – respiratory
    ("saans bilkul nahi aa rahi", "hi"), ("dam ghut gaya", "hi"),
    ("dam ghut raha hai", "hi"), ("sans band ho gayi", "hi"),
    ("दम घुट रहा है", "hi"),
    # Hindi – stroke
    ("laqwa pad gaya", "hi"), ("muh tedha ho gaya", "hi"),
    ("haath achanak kaam nahi kar raha", "hi"), ("bolna band ho gaya", "hi"),
    # Hindi – bleeding / shock
    ("khoon bahut bahut aa raha hai", "hi"), ("khoon band nahi ho raha", "hi"),
    ("behosh ho gaya", "hi"),
    ("होश नहीं है", "hi"),
    # Hindi – active suicide
    ("marne ki koshish kar raha hoon", "hi"), ("jaan de doonga", "hi"),
    ("khud ko maar loonga", "hi"),
    # Hindi – obstetric
    ("bachcha hil nahi raha kal se", "hi"), ("pani toot gaya", "hi"),
    ("prasav mein bahut khoon", "hi"),
    # Hindi – snakebite / envenomation (ALWAYS Tier 1)
    ("saanp ne kata abhi", "hi"), ("saanp ne kata", "hi"),
    ("saanp ka dank", "hi"), ("bichchu ne kata", "hi"),
    ("साँप ने काटा", "hi"), ("सांप ने काटा", "hi"),
    ("साँप का डंक", "hi"), ("सांप का डंक", "hi"), ("बिच्छू ने काटा", "hi"),
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

# Tier 2 = URGENT (LLM confirm allowed, EXCEPT crisis and snakebite)
TIER2_PHRASES: list[tuple[str, str]] = [
    ("chest pain", "en"), ("chest tightness", "en"),
    ("shortness of breath", "en"), ("short of breath", "en"),
    ("difficulty breathing", "en"), ("trouble breathing", "en"),
    ("severe headache", "en"), ("sudden confusion", "en"),
    ("seizure", "en"),
    # Note: "fitting" is intentionally NOT here — too common a word (fitting room etc.)
    # Use "fit" in medical context below instead
    ("having a fit", "en"),
    ("severe bleeding", "en"), ("poisoning", "en"),
    ("want to die", "en"), ("kill myself", "en"), ("suicide", "en"),
    ("self harm", "en"), ("no reason to live", "en"),
    ("labour pain", "en"), ("water broke", "en"),
    ("seene mein dard", "hi"), ("सीने में दर्द", "hi"),
    ("saans nahi aa rahi", "hi"), ("saans lene mein takleef", "hi"),
    ("behosh", "hi"), ("hosh nahi", "hi"),
    ("marna chahta hoon", "hi"), ("marna chahti hoon", "hi"),
    ("jaan dena chahta", "hi"),
    ("prasav peeda", "hi"), ("bachcha hil nahi raha", "hi"),
    ("seena mein dard ba", "bh"), ("dam naikhe", "bh"),
    ("bukhe byatha", "bn"), ("shash nite koshto", "bn"),
    ("நெஞ்சு வலி", "ta"), ("மூச்சு சிரமம்", "ta"),
    ("ఛాతీ నొప్పి", "te"), ("శ్వాస తీసుకోవడం కష్టం", "te"),
]

# Crisis / self-harm — NEVER downgraded by LLM confirm; always fires tier-2 minimum
CRISIS_PHRASES: list[str] = [
    # English
    "suicide", "want to die", "kill myself", "self harm", "no reason to live",
    "end my life", "hurt myself", "hurting myself",
    # Hindi – Devanagari
    "आत्महत्या", "मरना चाहता हूँ", "मरना चाहती हूँ", "खुद को नुकसान",
    # Hindi – transliteration
    "aatmahatya", "marna chahta hoon", "marna chahti hoon",
    "jaan dena chahta", "khud ko nuksaan",
]

# Words that appear in information / hypothetical questions rather than emergencies.
# Detecting these turns a potential tier-1 into info_query=True (show banner + still answer).
_INFO_QUESTION_PREFIXES = (
    "what is", "what are", "what happens", "what does", "what should",
    "how do", "how does", "how is", "how can",
    "tell me about", "explain", "describe",
    "what is an overdose", "what is a seizure", "what is eclampsia",
    "define ", "meaning of", "symptoms of",
    "can you tell me", "i want to know", "i want to learn",
    "is it normal", "is it possible", "what to do", "how to treat",
)

# Strong past-tense markers (specific time references — NOT short phrases like "had a")
_STRONG_PAST_MARKERS = (
    "last year", "last month", "years ago", "months ago",
    "pichle saal", "pehle hua tha", "kai saal pehle",
    "some years ago", "a few years ago", "a long time ago",
    "weeks ago", "last week",
)

# ---------------------------------------------------------------------------
# SNAKEBITE TOKEN CO-OCCURRENCE
# Devanagari-safe: split on whitespace + punctuation, not \W+
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"[\s,;:.!?।॥]+")

_SNAKE_A = frozenset([
    "saanp", "samp", "snake", "naag", "nag", "saanph",
    "साँप", "सांप", "नाग",          # Devanagari
])
_SNAKE_B = frozenset([
    "kata", "kaat", "kaata", "daas", "dank", "bite", "bit",
    "काटा", "काट", "डसा", "दंश",    # Devanagari
])

def _snake_cooccur(tokens: list[str]) -> bool:
    """True when at least one snake token AND one bite token appear."""
    has_a = any(t in _SNAKE_A for t in tokens)
    has_b = any(t in _SNAKE_B for t in tokens)
    return has_a and has_b


# ---------------------------------------------------------------------------
# SNAKEBITE FIRST-AID (MoHFW / WHO guidance — no LLM required)
# ---------------------------------------------------------------------------
SNAKEBITE_FIRST_AID = (
    "🐍 **SNAKEBITE — Call 112 / 108 immediately.**\n\n"
    "**What to do RIGHT NOW (WHO / MoHFW guidance):**\n"
    "1. Keep the person still and calm. Movement spreads venom faster.\n"
    "2. Immobilise the bitten limb — splint it at or below heart level.\n"
    "3. Remove rings, bangles, tight clothing near the bite site.\n"
    "4. Do NOT cut, suck, or squeeze the bite wound.\n"
    "5. Do NOT apply a tourniquet or ice.\n"
    "6. Note the snake's appearance if safe to do so — helps identify anti-venom.\n"
    "7. Rush to the nearest government hospital with anti-venom stock.\n\n"
    "📞 [Call 112](tel:112)  ·  🚑 [Call 108](tel:108)\n\n"
    "⚠️ Anti-venom is the only effective treatment. Get to hospital NOW."
)


# ---------------------------------------------------------------------------
# PUBLIC CLASSIFIER
# ---------------------------------------------------------------------------

def classify(text: str) -> Decision:
    """
    Pure-Python emergency classifier.  No Streamlit, no LLM.

    Logic order
    -----------
    1. Normalise text.
    2. Detect information questions → info_query (banner but answer).
    3. Snakebite co-occurrence (tier 1, snakebite=True, no veto ever).
    4. Tier-1 phrase scan.
    5. Crisis phrase scan (tier 2 minimum, crisis=True, no veto ever).
    6. Tier-2 phrase scan.
    """
    if not text or not text.strip():
        return Decision()

    # --- normalise ---
    nfkc = unicodedata.normalize("NFKC", text)   # normalise Unicode combining chars
    lower = nfkc.lower()
    tokens = [t for t in _SPLIT_RE.split(lower) if t]   # whitespace+punct split, Devanagari-safe

    # --- information / hypothetical question ---
    if any(lower.startswith(p) for p in _INFO_QUESTION_PREFIXES):
        # Still scan for tier, but flag as info_query so caller can add escape tap
        info_q = True
    else:
        info_q = False

    # --- snakebite token co-occurrence (tier 1, no veto) ---
    if _snake_cooccur(tokens):
        return Decision(tier=1, matched="snakebite (token co-occurrence)",
                        snakebite=True, info_query=info_q)

    # --- tier-1 phrase scan ---
    _SB_WORDS = ("snake", "snakebite", "saanp", "bichchu", "scorpion", "katlis", "dank", "डंक", "बिच्छू", "साँप", "सांप")
    for phrase, _lang in TIER1_PHRASES:
        if phrase.lower() in lower:
            is_sb = any(w in phrase.lower() for w in _SB_WORDS)
            return Decision(tier=1, matched=phrase, snakebite=is_sb, info_query=info_q)

    # --- crisis phrases (tier 2 minimum, never downgraded) ---
    for phrase in CRISIS_PHRASES:
        if phrase.lower() in lower:
            return Decision(tier=2, matched=phrase, crisis=True, info_query=info_q)

    # --- tier-2 phrase scan ---
    for phrase, _lang in TIER2_PHRASES:
        if phrase.lower() in lower:
            is_sb = any(w in phrase.lower() for w in _SB_WORDS)
            if is_sb:
                return Decision(tier=1, matched=phrase, snakebite=True, info_query=info_q)
            return Decision(tier=2, matched=phrase, snakebite=False, info_query=info_q)

    return Decision(info_query=info_q)


def is_strong_past_tense(text: str) -> bool:
    """
    True only for STRONG past-tense markers (specific time references).
    Short phrases like 'had a' do NOT qualify — 'he had a seizure and is not waking up'
    is a live emergency.
    """
    lower = text.lower()
    return any(marker in lower for marker in _STRONG_PAST_MARKERS)


def is_devanagari(text: str) -> bool:
    """True if text contains Devanagari characters."""
    return bool(re.search(r"[\u0900-\u097F]", text))
