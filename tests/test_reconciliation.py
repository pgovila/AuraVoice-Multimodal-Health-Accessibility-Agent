"""
tests/test_reconciliation.py — Tests for Indian prescription shorthand parser,
reconciliation table rendering, crash prevention, and spoken text generation.
"""

import pytest
from app import (
    parse_indian_prescription_shorthand,
    _spoken_reconciliation,
    _render_reconciliation,
)


class TestIndianPrescriptionShorthand:
    def test_shorthand_1_0_1(self):
        text = "1. Tab Metformin 500mg 1-0-1 after food"
        meds = parse_indian_prescription_shorthand(text)
        assert len(meds) == 1
        m = meds[0]
        assert "Metformin" in m["name"]
        assert "500mg" in m["dose"]
        assert m["schedule"]["morning"] is True
        assert m["schedule"]["afternoon"] is False
        assert m["schedule"]["night"] is True
        assert "1-0-1" in m["timing"]

    def test_shorthand_1_1_1(self):
        text = "Paracetamol 650mg 1-1-1"
        meds = parse_indian_prescription_shorthand(text)
        assert len(meds) == 1
        m = meds[0]
        assert m["schedule"]["morning"] is True
        assert m["schedule"]["afternoon"] is True
        assert m["schedule"]["night"] is True

    def test_shorthand_od_bd_tds_hs_sos(self):
        text = """
        Amlodipine 5mg OD
        Augmentin 625mg BD PC
        Pantoprazole 40mg OD HS AC
        Combiflam 400mg SOS
        """
        meds = parse_indian_prescription_shorthand(text)
        assert len(meds) == 4

        # Amlodipine OD -> morning
        assert meds[0]["schedule"]["morning"] is True
        assert meds[0]["schedule"]["night"] is False

        # Augmentin BD -> morning + night, after food (PC)
        assert meds[1]["schedule"]["morning"] is True
        assert meds[1]["schedule"]["night"] is True
        assert "PC" in meds[1]["timing"] or "after food" in meds[1]["timing"]

        # Pantoprazole OD HS -> night / bedtime
        assert meds[2]["schedule"]["night"] is True

        # Combiflam SOS -> as needed
        assert "SOS" in meds[3]["timing"] or "as needed" in meds[3]["timing"]

    def test_shorthand_sos_no_dose_rejected(self):
        text = "Combiflam SOS"
        meds = parse_indian_prescription_shorthand(text)
        assert len(meds) == 0

    def test_all_input_medicines_captured(self):
        text = """
        Atorvastatin 10mg 0-0-1 HS
        Telmisartan 40mg 1-0-0
        Eco-aspirin 75mg 0-1-0
        """
        meds = parse_indian_prescription_shorthand(text)
        assert len(meds) == 3
        names = [m["name"].lower() for m in meds]
        assert any("atorvastatin" in n for n in names)
        assert any("telmisartan" in n for n in names)
        assert any("aspirin" in n for n in names)


class TestReconciliationCrashPrevention:
    def test_null_schedule_and_flags_do_not_crash(self):
        """Simulate LLM returning null for schedule, flags, and warnings."""
        null_data = {
            "medicines": [
                {
                    "name": "Metformin 500mg",
                    "dose": "500mg",
                    "timing": None,
                    "schedule": None,  # explicitly null
                    "warnings": None,
                    "beers_flag": None,
                    "interaction_flags": None,
                    "duplicate_of": None,
                    "jan_aushadhi_note": None,
                }
            ],
            "unclear_timings": None,
            "disclaimer": None,
        }
        # Should not raise AttributeError or TypeError
        rendered = _render_reconciliation(null_data)
        assert "Metformin" in rendered
        assert "Medication Schedule" in rendered

    def test_spoken_reconciliation_builds_prose(self):
        """Verify spoken summary speaks medicine names, doses, and schedules."""
        data = {
            "medicines": [
                {
                    "name": "Metformin",
                    "dose": "500mg",
                    "schedule": {"morning": True, "afternoon": False, "evening": False, "night": True},
                    "timing": "1-0-1",
                },
                {
                    "name": "Amlodipine",
                    "dose": "5mg",
                    "schedule": {"morning": True, "afternoon": False, "evening": False, "night": False},
                    "timing": "OD",
                }
            ]
        }
        spoken = _spoken_reconciliation(data)
        assert "Metformin 500mg" in spoken
        assert "morning and night" in spoken
        assert "Amlodipine 5mg" in spoken
        assert "morning" in spoken
