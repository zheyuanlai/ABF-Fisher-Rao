"""The launch guard that Amendment 15.3's preflight did not have.

On 2026-08-15 the NaCl map ran 30 minutes on GPU 3 while Amendment 16.4 assigned that device to
C60. Every fact-based check passed -- the device was idle, the tree was clean, the pin matched --
because none of them was a check on the RULE. These tests pin the behaviour that would have
stopped it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import nacl_preflight as pf


def test_governing_clause_is_the_highest_numbered_not_the_first():
    """15.4 granted NaCl GPU 3; 16.4 took it back. The guard must read the later one."""
    text = (
        "#### 15.4 Compute: two GPUs for NaCl\n"
        "Once methane vacates GPU 3, NaCl may use GPU 3 as well as GPU 2.\n"
        "#### 16.4 Compute: GPU 3 reassigned\n"
        "C60 takes GPU 3; NaCl keeps GPU 2.\n")
    (_, num, _, body) = pf.governing_compute_clause(text)
    assert num == "16.4"
    assert pf.GRANT.findall(body) == ["2"], "must read 16.4's grant, not 15.4's"


def test_grant_parses_across_a_newline():
    """Regression: the real 16.4 wraps as 'NaCl\\nkeeps GPU 2'. A [^.\\n] class fails to parse it
    and the guard then reports 'no grant found' -- which refuses, but for the wrong reason, and
    would refuse the PERMITTED device too."""
    assert pf.GRANT.findall("**C60 takes GPU 3; NaCl\nkeeps GPU 2.**") == ["2"]


def test_grant_does_not_span_a_sentence_boundary():
    """A grant must not be assembled out of two unrelated sentences."""
    assert not pf.GRANT.findall("NaCl is paused. C60 takes GPU 3.")


def test_deny_detects_assignment_to_another_study():
    assert ("C60", "3") in [(w, d) for w, d in pf.DENY.findall("C60 takes GPU 3; NaCl keeps GPU 2.")]


def test_real_preregistration_grants_nacl_gpu2_and_refuses_gpu3():
    """Against the actual frozen document, not a fixture."""
    got = pf.governing_compute_clause()
    assert got is not None, "no GPU-allocating amendment found in the preregistration"
    (_, num, _, body) = got
    granted = sorted({int(g) for g in pf.GRANT.findall(body)})
    assert granted == [2], f"clause {num} grants NaCl {granted}, expected [2]"
    denied = {int(d) for w, d in pf.DENY.findall(body) if w.lower() != "nacl"}
    assert 3 in denied, f"clause {num} must record GPU 3 as another study's"


def test_unparseable_rule_refuses_rather_than_permits():
    """The failure being guarded against is a check that stayed SILENT when it did not apply.
    A clause the parser cannot read must never come back as permission."""
    text = ("#### 16.4 Compute\n"
            "Devices are allocated per the table in the appendix for GPU 2 and others.\n")
    (_, _, _, body) = pf.governing_compute_clause(text)
    assert not pf.GRANT.findall(body), "must not infer a grant from a clause it cannot parse"
