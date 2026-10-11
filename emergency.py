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
# -----------------------------------------------------------------# Tier 1 = CALL NOW — unambiguous life-threat
TIER1_PHRASES: list[tuple[str, str]] = [
    # English – cardiac
    ("heart attack", "en"), ("cardiac arrest", "en"),
    ("crushing chest pain", "en"), ("chest pain radiating", "en"),
    ("pain in left arm", "en"), ("chest is being crushed", "en"),
    ("chest crushing", "en"), ("collapsed suddenly", "en"),
    ("collapsed and not breathing", "en"), ("heart has stopped", "en"),
    ("heart stopped", "en"),
    # English – respiratory arrest
    ("not breathing", "en"), ("stopped breathing", "en"),
    ("cannot breathe at all", "en"), ("can't breathe at all", "en"),
    ("cant breathe at all", "en"), ("unable to breathe", "en"),
    ("airway blocked", "en"), ("choking", "en"),
    ("gasping for breath", "en"), ("gasping for air", "en"),
    ("lips turning blue", "en"), ("turning blue", "en"),
    # English – stroke (FAST)
    ("face drooping", "en"), ("face is drooping", "en"),
    ("arm weakness suddenly", "en"), ("arm weakness", "en"),
    ("speech slurred", "en"), ("sudden vision loss", "en"),
    ("worst headache of my life", "en"), ("sudden paralysis", "en"),
    ("cannot speak suddenly", "en"), ("unable to speak or move", "en"),
    # English – severe bleeding / shock
    ("bleeding won't stop", "en"), ("bleeding profusely", "en"),
    ("bleeding uncontrollably", "en"), ("massive blood loss", "en"),
    ("unconscious", "en"), ("unresponsive", "en"), ("not waking up", "en"),
    ("loss of consciousness", "en"), ("passed out and not waking up", "en"),
    ("no pulse", "en"), ("cannot feel his pulse", "en"), ("cannot feel a pulse", "en"),
    ("vomiting blood", "en"), ("coughing up blood", "en"),
    ("stab wound", "en"), ("gunshot", "en"),
    # English – anaphylaxis
    ("anaphylaxis", "en"), ("anaphylactic shock", "en"),
    ("throat is swelling shut", "en"), ("throat closing up", "en"),
    # English – poisoning / overdose
    ("overdose", "en"), ("swallowed poison", "en"), ("took too many pills", "en"),
    ("drank poison", "en"), ("swallowed chemical", "en"), ("consumed pesticide", "en"),
    # English – seizure ongoing
    ("seizure won't stop", "en"), ("convulsions", "en"),
    # English – obstetric
    ("eclampsia", "en"), ("baby not moving for hours", "en"),
    ("heavy bleeding in pregnancy", "en"),
    ("heavy bleeding in third trimester", "en"),
    ("third trimester heavy bleeding", "en"),
    ("placenta bleeding", "en"),
    # English – envenomation (snakebite / scorpion sting are ALWAYS Tier 1)
    ("bitten by snake", "en"), ("bitten by a snake", "en"),
    ("snakebite", "en"), ("snake bite", "en"),
    ("scorpion sting severe", "en"), ("scorpion sting", "en"),
    # Hindi – cardiac
    ("dil ka daura pad gaya", "hi"), ("dil ka dora", "hi"),
    ("seene mein bahut tez dard", "hi"), ("chhati mein tez dard", "hi"),
    ("seene mein dard aa raha hai", "hi"),
    ("dil ki dhadkan ruk gayi", "hi"), ("dil ruk gaya", "hi"),
    ("दिल का दौरा", "hi"), ("सीने में बहुत तेज दर्द", "hi"),
    ("दिल की धड़कन रुक गई", "hi"),
    # Hindi – respiratory
    ("saans bilkul nahi aa rahi", "hi"), ("dam ghut gaya", "hi"),
    ("dam ghut raha hai", "hi"), ("sans band ho gayi", "hi"),
    ("saans achanak band ho gayi", "hi"),
    ("दम घुट रहा है", "hi"), ("गला घुट रहा है", "hi"),
    # Hindi – stroke
    ("laqwa pad gaya", "hi"), ("muh tedha ho gaya", "hi"),
    ("haath achanak kaam nahi kar raha", "hi"), ("bolna band ho gaya", "hi"),
    ("achanak paralysis", "hi"), ("lakwa mar gaya", "hi"),
    # Hindi – bleeding / shock
    ("khoon bahut bahut aa raha hai", "hi"), ("khoon band nahi ho raha", "hi"),
    ("behosh ho gaya", "hi"), ("achanak behosh hokar gir", "hi"),
    ("hosh mein nahi aa raha", "hi"),
    ("होश नहीं है", "hi"),
    ("muh se khoon aa raha hai", "hi"), ("khoon ki ulti", "hi"),
    ("खून की उल्टी", "hi"),
    # Hindi – poisoning / overdose
    ("zeher khaa liya", "hi"), ("zehar pee liya", "hi"),
    ("ज़हर खा लिया", "hi"),
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
    ("বুকে অনেক ব্যথা", "bn"), ("শ্বাস নিতে পারছি না एकदम", "bn"),
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
    ("breathlessness", "en"), ("breathless", "en"),
    ("severe asthma", "en"), ("asthma attack", "en"),
    ("severe headache", "en"), ("sudden confusion", "en"),
    ("seizure", "en"),
    # Note: "fitting" is intentionally NOT here — too common a word (fitting room etc.)
    # Use "fit" in medical context below instead
    ("having a fit", "en"),
    ("fainted", "en"), ("fainting repeatedly", "en"), ("blacking out", "en"),
    ("sudden severe dizziness", "en"),
    ("severe abdominal pain", "en"), ("severe stomach pain", "en"),
    ("appendicitis", "en"), ("acute belly pain", "en"),
    ("high fever with stiff neck", "en"), ("fever with confusion", "en"),
    ("irregular pulse", "en"), ("racing heart and dizzy", "en"),
    ("diabetic ketoacidosis", "en"),
    ("severe bleeding", "en"), ("poisoning", "en"),
    ("want to die", "en"), ("kill myself", "en"), ("suicide", "en"),
    ("self harm", "en"), ("no reason to live", "en"),
    ("not want to live", "en"), ("dont want to live", "en"),
    ("do not want to live", "en"), ("tired of living", "en"),
    ("give up on life", "en"), ("wish i was dead", "en"), ("wish i were dead", "en"),
    ("labour pain", "en"), ("water broke", "en"),
    ("seene mein dard", "hi"), ("सीने में दर्द", "hi"),
    ("saans nahi aa rahi", "hi"), ("saans lene mein takleef", "hi"),
    ("saans lene mein dikkat", "hi"), ("सांस लेने में दिक्कत", "hi"),
    ("saans phool rahi hai", "hi"), ("सांस फूल रही है", "hi"),
    ("pet mein asahaniya dard", "hi"), ("पेट में बहुत तेज दर्द", "hi"),
    ("gardhan mein akad", "hi"), ("gardhan akad gayi", "hi"),
    ("dhadkan bahut tez", "hi"), ("dhadkan aniyamit", "hi"),
    ("chakkar aakar gir gaya", "hi"), ("चक्कर आकर गिर", "hi"),
    ("dama ka daura", "hi"), ("दमा का दौरा", "hi"),
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
    "suicide", "suicidal", "want to die", "kill myself", "going to kill myself",
    "about to end my life", "pills to die", "taking pills to die",
    "self harm", "no reason to live", "end my life", "hurt myself", "hurting myself",
    "not want to live", "dont want to live", "don't want to live", "do not want to live",
    "tired of living", "give up on life", "better off dead", "wish i were dead", "wish i was dead",
    "end it all", "take my own life", "cannot go on living", "can't go on living",
    # Hindi – Devanagari
    "आत्महत्या", "मरना चाहता हूँ", "मरना चाहती हूँ", "खुद को नुकसान", "खुदकुशी",
    "जीना नहीं चाहता", "जीना नहीं चाहती", "जीने की इच्छा नहीं", "जान देना चाहता",
    # Hindi – transliteration
    "aatmahatya", "marna chahta hoon", "marna chahti hoon", "marna chahta", "marna chahti",
    "jaan dena chahta", "jaan de doonga", "khud ko nuksaan", "khudkushi",
    "marne ki koshish", "khud ko maar loonga",
    "jeena nahi chahta", "jeena nahi chahti", "jeene ka mann nahi", "zindagi se tang",
    # Regional
    "marab chahta",
]

# Words that appear in information / hypothetical questions rather than emergencies.
# Detecting these turns a potential tier-1 into info_query=True (show banner + still answer)
# ONLY when no live acute cue is present in the text.
_INFO_QUESTION_PREFIXES = (
    "what is", "what are", "what happens", "what does",
    "how does", "how is",
    "tell me about", "explain", "describe",
    "what is an overdose", "what is a seizure", "what is eclampsia",
    "define ", "meaning of", "symptoms of",
    "can you tell me", "i want to know", "i want to learn",
    "is it normal", "is it possible", "how to treat",
)

# Live cues: regex word-boundary match so substrings like "anatomy" do not falsely trigger "my"
_LIVE_CUES_RE = re.compile(
    r"\b(?:my|his|her|he\s+is|she\s+is|right\s+now|abhi)\b",
    re.IGNORECASE,
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
# SNAKEBITE FIRST-AID (MoHFW / WHO guidance — verified medical source links)
# ---------------------------------------------------------------------------
SNAKEBITE_FIRST_AID = (
    "🐍 **SNAKEBITE FIRST AID — Call 112 / 108 immediately.**\n\n"
    "**Immediate actions (WHO & Ministry of Health & Family Welfare guidelines):**\n"
    "1. Keep the patient completely still and calm (movement accelerates venom spread).\n"
    "2. Immobilise the bitten limb — splint it at or slightly below heart level.\n"
    "3. Remove constricting items (rings, bangles, tight clothes, watches) near the bite.\n"
    "4. Do NOT cut, incise, suck, or squeeze the bite wound.\n"
    "5. Do NOT apply a tourniquet, herbal paste, chemicals, or ice.\n"
    "6. Note the snake's physical appearance from a safe distance (assists anti-snake venom selection).\n"
    "7. Rush immediately to the nearest government hospital equipped with Anti-Snake Venom (ASV).\n\n"
    "📞 [Call 112](tel:112)  ·  🚑 [Call 108](tel:108)\n\n"
    "⚠️ Anti-snake venom (ASV) is the only proven medical antidote. Do not delay transport.\n\n"
    "📖 **Verified Medical Sources:**\n"
    "- [WHO Guidelines for the Clinical Management of Snakebites](https://www.who.int/publications/i/item/9789290225300)\n"
    "- [MoHFW National Action Plan for Prevention & Control of Snakebite Envenoming (NAP-SE)](https://dghs.gov.in/content/1376_3_SnakeBite.aspx)"
)


# ---------------------------------------------------------------------------
# PUBLIC CLASSIFIER
# ---------------------------------------------------------------------------

def classify(text: str) -> Decision:
    """
    Pure-Python emergency classifier. No Streamlit, no LLM.

    Logic order
    -----------
    1. Normalise text.
    2. Detect information questions (only when NO live cue is present).
    3. Snakebite co-occurrence (tier 1, snakebite=True, no veto ever).
    4. Crisis phrase scan (tier 2, crisis=True, scanned before tier 1 so self-harm routes correctly).
    5. Tier-1 phrase scan.
    6. Tier-2 phrase scan.
    """
    if not text or not text.strip():
        return Decision()

    # --- normalise ---
    nfkc = unicodedata.normalize("NFKC", text)   # normalise Unicode combining chars
    lower = nfkc.lower()
    tokens = [t for t in _SPLIT_RE.split(lower) if t]   # whitespace+punct split, Devanagari-safe

    # --- information / hypothetical question ---
    has_live_cue = bool(_LIVE_CUES_RE.search(lower))
    if not has_live_cue and any(lower.startswith(p) for p in _INFO_QUESTION_PREFIXES):
        info_q = True
    else:
        info_q = False

    # --- snakebite token co-occurrence (tier 1, no veto) ---
    if _snake_cooccur(tokens):
        return Decision(tier=1, matched="snakebite (token co-occurrence)",
                        snakebite=True, info_query=info_q)

    # --- crisis phrases (tier 2 minimum, never downgraded, scanned BEFORE tier-1) ---
    for phrase in CRISIS_PHRASES:
        if phrase.lower() in lower:
            return Decision(tier=2, matched=phrase, crisis=True, info_query=info_q)

    # --- tier-1 phrase scan ---
    _SB_WORDS = ("snake", "snakebite", "saanp", "bichchu", "scorpion", "katlis", "dank", "डंक", "बिच्छू", "साँप", "सांप")
    for phrase, _lang in TIER1_PHRASES:
        if phrase.lower() in lower:
            is_sb = any(w in phrase.lower() for w in _SB_WORDS)
            return Decision(tier=1, matched=phrase, snakebite=is_sb, info_query=info_q)

    # --- tier-2 phrase scan ---
    for phrase, _lang in TIER2_PHRASES:
        if phrase.lower() in lower:
            is_sb = any(w in phrase.lower() for w in _SB_WORDS)
            if is_sb:
                return Decision(tier=1, matched=phrase, snakebite=True, info_query=info_q)
            return Decision(tier=2, matched=phrase, snakebite=False, info_query=info_q)

    return Decision(info_query=info_q)


_PAST_TIME_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|several|a few|some|many)\s+(?:years?|months?|weeks?|days?)\s+ago\b|"
    r"\b(?:pichle|kai|kuch|do|teen|chaar|paanch|ek)\s+(?:saal|mahine|hafte|din)\s+(?:pehle|purana)\b|"
    r"(?:साल|महीने|हफ़्ते|दिन)\s+पहले",
    re.IGNORECASE,
)


def is_strong_past_tense(text: str) -> bool:
    """
    True only for STRONG past-tense markers (specific time references).
    Short phrases like 'had a' do NOT qualify — 'he had a seizure and is not waking up'
    is a live emergency.
    """
    lower = text.lower()
    if any(marker in lower for marker in _STRONG_PAST_MARKERS):
        return True
    return bool(_PAST_TIME_RE.search(lower))


def is_devanagari(text: str) -> bool:
    """True if text contains Devanagari characters."""
    return bool(re.search(r"[\u0900-\u097F]", text))


def resolve_rule_tier(dec: Decision, text: str, confirmed_past: bool = False) -> int:
    """
    Canonical post-classification decision resolver.
    Shared across eval_emergency.py and app.py to prevent logic drift.
    Returns:
      -1 : Pending past-tense verification (live check in app)
       0 : Non-alert (confirmed past event, info query, or tier 0)
       1 : Tier 1 (CALL NOW)
       2 : Tier 2 (URGENT / Crisis)
    """
    if dec.crisis:
        return max(dec.tier, 2)
    if dec.tier == 1 and not confirmed_past and is_strong_past_tense(text):
        return -1
    if (dec.tier > 0 and confirmed_past) or (dec.tier == 2 and is_strong_past_tense(text)):
        return 0
    if dec.info_query:
        return 0
    return dec.tier


def evaluate_rule_prediction(text: str, confirmed_past: bool = False) -> int:
    """
    Canonical rule prediction mapping for evaluation harness.
    Maps pending past-tense sentinel (-1) to non-alert (0).
    """
    dec = classify(text)
    tier = resolve_rule_tier(dec, text, confirmed_past=confirmed_past)
    return 0 if tier < 0 else tier
