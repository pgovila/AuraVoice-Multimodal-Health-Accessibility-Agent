"""
tests/test_emergency.py
=======================
Independent pytest suite for emergency.py classify().

ALL test sentences were written independently — none are verbatim copies of
the seed phrase lists. This tests generalisation, not memorisation.

Run:
    pytest tests/test_emergency.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from emergency import classify, is_strong_past_tense, Decision


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tier(text: str) -> int:
    return classify(text).tier

def is_crisis(text: str) -> bool:
    return classify(text).crisis

def is_snakebite(text: str) -> bool:
    return classify(text).snakebite

def is_info(text: str) -> bool:
    return classify(text).info_query


# ---------------------------------------------------------------------------
# TIER-1  (must never return 0)
# ---------------------------------------------------------------------------

class TestTier1English:
    def test_cardiac_paraphrase(self):
        assert tier("My chest is being crushed, pain into my left arm") >= 1

    def test_cardiac_direct(self):
        assert tier("I am having a heart attack") >= 1

    def test_not_breathing(self):
        assert tier("she has stopped breathing completely") >= 1

    def test_stroke_face(self):
        assert tier("his face is drooping and arm is weak") >= 1

    def test_unconscious_now(self):
        assert tier("he is unconscious on the floor right now") >= 1

    def test_overdose_active(self):
        assert tier("she took an overdose of pills") >= 1

    def test_seizure_ongoing(self):
        assert tier("convulsions have been going on for 10 minutes") >= 1

    def test_no_pulse(self):
        assert tier("we cannot feel his pulse and he is not responding") >= 1

    def test_snakebite_explicit_en(self):
        d = classify("I was bitten by a snake on my ankle")
        assert d.tier >= 1
        assert d.snakebite is True

    def test_bleeding_profuse(self):
        assert tier("she is bleeding profusely from the wound") >= 1


class TestTier1Hindi:
    def test_cardiac_devanagari(self):
        assert tier("दिल का दौरा पड़ रहा है") >= 1

    def test_respiratory_transliteration(self):
        assert tier("saans bilkul nahi aa rahi, dam ghut raha hai") >= 1

    def test_unconscious_hindi(self):
        assert tier("behosh ho gaya hai, uthh nahi raha") >= 1

    def test_snakebite_hindi_now(self):
        d = classify("saanp ne kata abhi haath pe")
        assert d.tier >= 1
        assert d.snakebite is True


class TestTier1SnakebiteCooccurrence:
    """Token co-occurrence — indirect phrasings that don't match a literal phrase."""

    def test_indirect_hindi_transliteration(self):
        d = classify("koi cheez ne kata, shayad saanp tha")
        assert d.tier >= 1
        assert d.snakebite is True

    def test_english_tokens(self):
        d = classify("something in the grass bit me, think it was a snake")
        assert d.tier >= 1
        assert d.snakebite is True

    def test_devanagari_tokens(self):
        d = classify("साँप ने काट लिया")
        assert d.tier >= 1
        assert d.snakebite is True

    def test_snakebite_is_tier1_not_tier2(self):
        """Any snakebite should be tier 1, not 2."""
        d = classify("snake bite happened just now")
        assert d.tier == 1
        assert d.snakebite is True


# ---------------------------------------------------------------------------
# TIER-2 (must be >= 1; LLM can veto to 0, but rule should fire)
# ---------------------------------------------------------------------------

class TestTier2English:
    def test_chest_tightness(self):
        assert tier("I have chest tightness climbing stairs") >= 2

    def test_shortness_of_breath(self):
        assert tier("I am very short of breath since this morning") >= 2

    def test_suicidal_ideation(self):
        d = classify("I want to die, there is no reason to live")
        assert d.tier >= 2
        assert d.crisis is True

    def test_self_harm(self):
        d = classify("I have been hurting myself")
        assert d.tier >= 2
        assert d.crisis is True


class TestCrisis:
    """Crisis phrases must never be tier 0 and must set crisis=True."""

    def test_suicide_en(self):
        d = classify("I am thinking about suicide")
        assert d.tier >= 2
        assert d.crisis is True

    def test_kill_myself_en(self):
        d = classify("I feel like I should kill myself")
        assert d.tier >= 2
        assert d.crisis is True

    def test_aatmahatya_devanagari(self):
        d = classify("मुझे आत्महत्या के विचार आ रहे हैं")
        assert d.tier >= 2
        assert d.crisis is True

    def test_marna_chahta_hoon(self):
        d = classify("marna chahta hoon, bahut thak gaya hoon")
        assert d.tier >= 2
        assert d.crisis is True

    def test_khud_ko_nuksaan(self):
        d = classify("mai khud ko nuksaan pahunchana chahta hoon")
        assert d.tier >= 2
        assert d.crisis is True

    def test_end_my_life(self):
        d = classify("I am going to end my life tonight")
        assert d.tier >= 2
        assert d.crisis is True


# ---------------------------------------------------------------------------
# TIER-0  (must not fire)
# ---------------------------------------------------------------------------

class TestNegatives:
    def test_metaphorical_heart(self):
        assert tier("my heart is full of happiness today") == 0

    def test_metaphorical_stroke(self):
        assert tier("I had a stroke of luck at the market") == 0

    def test_fitting_room(self):
        # "fitting" removed from TIER2; "fitting room" should not fire
        assert tier("the fitting room is on the left side of the store") == 0

    def test_having_a_fit_fires(self):
        # medical "having a fit" should fire tier 2
        assert tier("my child is having a fit") >= 2

    def test_routine_fever(self):
        assert tier("I have a mild fever since yesterday") == 0

    def test_medicine_question(self):
        assert tier("how should I take Metformin with food?") == 0

    def test_greeting(self):
        assert tier("hello, can you help me with my medicines?") == 0

    def test_no_chest_pain_hindi_still_fires(self):
        # Design decision: no negation suppression — a false positive is safer than a miss.
        # "seene mein dard" is in TIER2, so this fires even with "nahi".
        # The LLM confirm step (in app.py) can downgrade it to tier 0.
        assert tier("seene mein dard nahi hai aaj") >= 2   # intentional: rule fires, LLM vetoes

    def test_not_unconscious_context(self):
        # "unconscious" appears but in a medical study question
        d = classify("what is the difference between unconscious and comatose?")
        # Should fire tier-1 (unconscious is in T1) BUT as info_query
        assert d.info_query is True

    def test_past_event_not_past_tense_marker(self):
        # "had a" without a strong time marker → still live emergency heuristic
        # "he had a seizure and is not waking up" — MUST be tier 1
        assert tier("he had a seizure and is not waking up") >= 1


# ---------------------------------------------------------------------------
# INFORMATION QUERIES
# ---------------------------------------------------------------------------

class TestInfoQuery:
    def test_overdose_question(self):
        d = classify("what happens in an overdose?")
        assert d.info_query is True   # should show banner but still answer

    def test_seizure_question(self):
        d = classify("what does a seizure look like?")
        assert d.info_query is True

    def test_eclampsia_question(self):
        d = classify("what is eclampsia?")
        assert d.info_query is True

    def test_unconscious_question(self):
        d = classify("what is the difference between unconscious and comatose?")
        assert d.info_query is True


# ---------------------------------------------------------------------------
# STRONG PAST-TENSE MARKERS
# ---------------------------------------------------------------------------

class TestPastTense:
    def test_last_year(self):
        assert is_strong_past_tense("My father had a heart attack last year") is True

    def test_years_ago(self):
        assert is_strong_past_tense("I had a seizure years ago") is True

    def test_current_emergency_not_past(self):
        # "had a" alone is NOT a strong past marker
        assert is_strong_past_tense("he had a seizure and is not waking up") is False

    def test_no_marker(self):
        assert is_strong_past_tense("I have chest pain right now") is False

    def test_hindi_past_marker(self):
        assert is_strong_past_tense("pichle saal dil ka daura pada tha") is True


# ---------------------------------------------------------------------------
# REGRESSION — previous bugs that should never recur
# ---------------------------------------------------------------------------

class TestRegressions:
    def test_sample_snakebite_case_demo(self):
        """The demo sample case must return tier >= 1 with snakebite=True."""
        d = classify("mere haath par koi cheez ne kata, shayad saanp tha")
        assert d.tier >= 1
        assert d.snakebite is True

    def test_overdose_not_downgraded_as_info(self):
        """Active overdose report must not be treated as info query."""
        d = classify("she took an overdose of sleeping pills an hour ago")
        assert d.info_query is False
        assert d.tier >= 1

    def test_suicidal_text_not_downgraded(self):
        """Crisis phrases cannot be returned as tier 0."""
        d = classify("aatmahatya karna chahta hoon")
        assert d.tier >= 2
        assert d.crisis is True

    def test_devanagari_snakebite_tokens_match(self):
        """Devanagari tokens must be found (split on whitespace, not \\W+)."""
        d = classify("साँप ने मुझे काट लिया")
        assert d.snakebite is True
