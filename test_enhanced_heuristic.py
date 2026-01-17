#!/usr/bin/env python3
"""
Test script for enhanced stance-aware heuristic improvements.
Tests negation handling, intensity scoring, semantic matching, and temporal awareness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.tools.real_toolkit import RealToolkit

def test_negation_handling():
    """Test that negation is properly detected and handled."""
    print("=" * 60)
    print("TEST 1: Negation Handling")
    print("=" * 60)
    
    # Test case 1: "not confirmed" should NOT be treated as affirm cue
    text1 = "The claim was not confirmed by any experts."
    has_affirm = RealToolkit._has_affirm_cues(text1)
    print(f"Text: '{text1}'")
    print(f"Has affirm cues: {has_affirm}")
    print(f"Expected: False (negated)")
    print(f"✓ PASS" if not has_affirm else "✗ FAIL")
    print()
    
    # Test case 2: "never debunked" should NOT be treated as refute cue
    text2 = "This theory has never been debunked despite many attempts."
    has_refute = RealToolkit._has_refute_cues(text2)
    print(f"Text: '{text2}'")
    print(f"Has refute cues: {has_refute}")
    print(f"Expected: False (negated)")
    print(f"✓ PASS" if not has_refute else "✗ FAIL")
    print()
    
    # Test case 3: "confirmed" without negation should work
    text3 = "The event was confirmed by multiple independent sources."
    has_affirm = RealToolkit._has_affirm_cues(text3)
    print(f"Text: '{text3}'")
    print(f"Has affirm cues: {has_affirm}")
    print(f"Expected: True")
    print(f"✓ PASS" if has_affirm else "✗ FAIL")
    print()

def test_intensity_scoring():
    """Test stance intensity scoring."""
    print("=" * 60)
    print("TEST 2: Intensity Scoring")
    print("=" * 60)
    
    # Strong refutation
    text1 = "This claim has been proven false by scientific research."
    intensity1 = RealToolkit._compute_refute_intensity(text1)
    print(f"Text: '{text1}'")
    print(f"Refute intensity: {intensity1:.2f}")
    print(f"Expected: > 0.8 (strong)")
    print(f"✓ PASS" if intensity1 > 0.8 else "✗ FAIL")
    print()
    
    # Weak refutation
    text2 = "Some experts have questioned this claim."
    intensity2 = RealToolkit._compute_refute_intensity(text2)
    print(f"Text: '{text2}'")
    print(f"Refute intensity: {intensity2:.2f}")
    print(f"Expected: 0.3-0.5 (weak)")
    print(f"✓ PASS" if 0.3 <= intensity2 <= 0.5 else "✗ FAIL")
    print()
    
    # Strong affirmation
    text3 = "This fact has been verified and confirmed by scientific consensus."
    intensity3 = RealToolkit._compute_affirm_intensity(text3)
    print(f"Text: '{text3}'")
    print(f"Affirm intensity: {intensity3:.2f}")
    print(f"Expected: > 0.8 (strong)")
    print(f"✓ PASS" if intensity3 > 0.8 else "✗ FAIL")
    print()

def test_hoax_detection():
    """Test enhanced hoax claim detection."""
    print("=" * 60)
    print("TEST 3: Enhanced Hoax Detection")
    print("=" * 60)
    
    # Direct hoax marker
    claim1 = "The moon landing was a hoax filmed in a studio."
    is_hoax1 = RealToolkit._is_hoaxy_claim(claim1)
    print(f"Claim: '{claim1}'")
    print(f"Is hoax claim: {is_hoax1}")
    print(f"Expected: True")
    print(f"✓ PASS" if is_hoax1 else "✗ FAIL")
    print()
    
    # Semantic pattern
    claim2 = "The government is hiding information about this event."
    is_hoax2 = RealToolkit._is_hoaxy_claim(claim2)
    print(f"Claim: '{claim2}'")
    print(f"Is hoax claim: {is_hoax2}")
    print(f"Expected: True (semantic pattern)")
    print(f"✓ PASS" if is_hoax2 else "✗ FAIL")
    print()
    
    # Normal claim
    claim3 = "The 2024 Summer Olympics were held in Paris."
    is_hoax3 = RealToolkit._is_hoaxy_claim(claim3)
    print(f"Claim: '{claim3}'")
    print(f"Is hoax claim: {is_hoax3}")
    print(f"Expected: False")
    print(f"✓ PASS" if not is_hoax3 else "✗ FAIL")
    print()

def test_semantic_anchor_matching():
    """Test semantic anchor matching with fuzzy matching."""
    print("=" * 60)
    print("TEST 4: Semantic Anchor Matching")
    print("=" * 60)
    
    anchors = {"olympics", "2024", "paris"}
    
    # Exact matches
    blob1 = "The 2024 Olympics were held in Paris, France."
    matches1, score1 = RealToolkit._semantic_anchor_match(anchors, blob1)
    print(f"Anchors: {anchors}")
    print(f"Blob: '{blob1}'")
    print(f"Matches: {matches1}, Weighted score: {score1:.2f}")
    print(f"Expected: 3 matches, score > 4.0")
    print(f"✓ PASS" if matches1 == 3 and score1 > 4.0 else "✗ FAIL")
    print()
    
    # Fuzzy match (typo)
    blob2 = "The 2024 Olympic games were in Parris."  # typo: Parris
    matches2, score2 = RealToolkit._semantic_anchor_match(anchors, blob2)
    print(f"Anchors: {anchors}")
    print(f"Blob: '{blob2}'")
    print(f"Matches: {matches2}, Weighted score: {score2:.2f}")
    print(f"Expected: ~2.0 matches (fuzzy), score > 2.0")
    print(f"✓ PASS" if matches2 >= 1.5 and score2 > 2.0 else "✗ FAIL")
    print()

def test_temporal_consistency():
    """Test temporal consistency checking."""
    print("=" * 60)
    print("TEST 5: Temporal Consistency")
    print("=" * 60)
    
    # Consistent years
    claim_years1 = ["2024"]
    evidence_years1 = ["2024"]
    consistent1 = RealToolkit._check_temporal_consistency(claim_years1, evidence_years1)
    print(f"Claim years: {claim_years1}, Evidence years: {evidence_years1}")
    print(f"Consistent: {consistent1}")
    print(f"Expected: True")
    print(f"✓ PASS" if consistent1 else "✗ FAIL")
    print()
    
    # Inconsistent years (significant gap)
    claim_years2 = ["2024"]
    evidence_years2 = ["2020"]
    consistent2 = RealToolkit._check_temporal_consistency(claim_years2, evidence_years2)
    print(f"Claim years: {claim_years2}, Evidence years: {evidence_years2}")
    print(f"Consistent: {consistent2}")
    print(f"Expected: False (4 year gap)")
    print(f"✓ PASS" if not consistent2 else "✗ FAIL")
    print()
    
    # Close years (within tolerance)
    claim_years3 = ["2024"]
    evidence_years3 = ["2023"]
    consistent3 = RealToolkit._check_temporal_consistency(claim_years3, evidence_years3)
    print(f"Claim years: {claim_years3}, Evidence years: {evidence_years3}")
    print(f"Consistent: {consistent3}")
    print(f"Expected: True (within 2 year tolerance)")
    print(f"✓ PASS" if consistent3 else "✗ FAIL")
    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ENHANCED STANCE-AWARE HEURISTIC TEST SUITE")
    print("=" * 60 + "\n")
    
    test_negation_handling()
    test_intensity_scoring()
    test_hoax_detection()
    test_semantic_anchor_matching()
    test_temporal_consistency()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
