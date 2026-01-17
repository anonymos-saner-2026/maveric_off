

from src.tools.real_toolkit import RealToolkit

class TestAdvancedReasoning:
    
    # -------------------------------------------------------------------------
    # GAP 1: Numerical Reasoning Tests
    # -------------------------------------------------------------------------
    
    def test_number_extraction_simple(self):
        text = "The population of Paris is 2.2 million people."
        # Should extract 2.2 million
        nums = RealToolkit._extract_numbers_with_units(text)
        print(f"\nDEBUG: extracted {nums}")
        assert len(nums) == 1
        assert nums[0][1] == 2200000.0
        assert "population of paris" in nums[0][0] # Context

    def test_number_extraction_units(self):
        text = "Mount Everest is 8,849 meters tall."
        nums = RealToolkit._extract_numbers_with_units(text)
        assert len(nums) == 1
        assert nums[0][1] == 8849.0
        assert nums[0][2] == "meters"

    def test_number_extraction_feet_conversion(self):
        text = "Mount Everest is 29,032 feet tall."
        nums = RealToolkit._extract_numbers_with_units(text)
        assert len(nums) == 1
        # 29032 * 0.3048 = 8848.95
        assert abs(nums[0][1] - 8848.95) < 1.0 
        assert nums[0][2] == "meters"  # Normalized unit

    def test_numerical_comparison_tolerance(self):
        # 10% tolerance is default
        assert RealToolkit._compare_numerical_values(100, 105) == True
        assert RealToolkit._compare_numerical_values(100, 120) == False
        
    def test_numerical_verification_match(self):
        claim_nums = [("context", 2200000, "")]
        evidence_nums = [("context", 2161000, "")] # ~1.7% diff
        assert RealToolkit._verify_numerical_claim(claim_nums, evidence_nums) == True

    def test_numerical_verification_mismatch(self):
        claim_nums = [("context", 2200000, "")]
        evidence_nums = [("context", 500000, "")] # Huge diff
        assert RealToolkit._verify_numerical_claim(claim_nums, evidence_nums) == False

    # -------------------------------------------------------------------------
    # GAP 2: Comparative/Superlative Tests
    # -------------------------------------------------------------------------

    def test_detect_superlative(self):
        text = "Mount Everest is the highest mountain in the world."
        info = RealToolkit._detect_comparative_claim(text)
        assert info["type"] == "superlative"
        assert info["comparator"] == "highest"

    def test_detect_comparative(self):
        text = "Paris is larger than Lyon."
        info = RealToolkit._detect_comparative_claim(text)
        assert info["type"] == "comparative"
        assert info["entity_a"] == "paris"
        assert info["entity_b"] == "lyon"
        assert "larger" in info["comparator"]

    def test_verify_superlative_match(self):
        info = {"type": "superlative", "comparator": "highest"}
        evidence = "Mount Everest is Earth's highest peak at 8849m."
        assert RealToolkit._verify_comparative_claim(info, evidence) == True

    def test_verify_superlative_synonym(self):
        info = {"type": "superlative", "comparator": "tallest"}
        evidence = "Mount Everest is the highest mountain."
        assert RealToolkit._verify_comparative_claim(info, evidence) == True

    def test_verify_comparative_match(self):
        info = {
            "type": "comparative", 
            "entity_a": "paris", 
            "entity_b": "lyon", 
            "comparator": "larger"
        }
        evidence = "Paris is larger than Lyon by distinct population margin."
        assert RealToolkit._verify_comparative_claim(info, evidence) == True

    # -------------------------------------------------------------------------
    # GAP 3: Multi-hop Reasoning Tests
    # -------------------------------------------------------------------------

    def test_detect_multihop_possessive(self):
        text = "Obama's wife was born in Chicago."
        info = RealToolkit._detect_multi_hop_claim(text)
        assert info["type"] == "possessive"
        assert "possessives" in info["hint"]

    def test_detect_multihop_relation(self):
        text = "The author of Harry Potter lives in UK."
        info = RealToolkit._detect_multi_hop_claim(text)
        assert info["type"] == "relation"
        assert "author of" in info["hint"]

if __name__ == "__main__":
    t = TestAdvancedReasoning()
    t.test_number_extraction_simple()
    t.test_number_extraction_units()
    t.test_number_extraction_feet_conversion()
    t.test_numerical_comparison_tolerance()
    t.test_numerical_verification_match()
    t.test_numerical_verification_mismatch()
    t.test_detect_superlative()
    t.test_detect_comparative()
    t.test_verify_superlative_match()
    t.test_verify_superlative_synonym()
    t.test_verify_comparative_match()
    t.test_detect_multihop_possessive()
    t.test_detect_multihop_relation()
    print("All advanced reasoning tests passed!")
