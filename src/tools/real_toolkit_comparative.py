    # ---------- Comparative/Superlative Handling ----------
    @staticmethod
    def _detect_comparative_claim(text: str) -> Optional[Dict[str, Any]]:
        """
        Detect and parse comparative/superlative claims.
        Returns: {
            "type": "comparative" | "superlative",
            "entity": str,
            "attribute": str,
            "comparator": str,
            "reference": str    # comparison target (for comparative)
        }
        """
        text_lower = text.lower()
        
        # Superlative patterns
        # "X is the tallest/most/best..."
        sup_patterns = [
            r"the (most \w+|least \w+|best|worst|tallest|shortest|biggest|smallest|highest|lowest|fastest|slowest|oldest|youngest|first|last) (.+)",
        ]
        
        for p in sup_patterns:
            m = re.search(p, text_lower)
            if m:
                return {
                    "type": "superlative",
                    "comparator": m.group(1),
                    "context": m.group(2)
                }
        
        # Comparative patterns
        # "X is taller/more than Y"
        comp_patterns = [
            r"(.+) is (more \w+|less \w+|better|worse|\w+er) than (.+)",
        ]
        
        for p in comp_patterns:
            m = re.search(p, text_lower)
            if m:
                return {
                    "type": "comparative",
                    "entity_a": m.group(1).strip(),
                    "comparator": m.group(2).strip(),
                    "entity_b": m.group(3).strip()
                }
        
        return None

    @staticmethod
    def _verify_comparative_claim(
        claim_info: Dict[str, Any],
        evidence_text: str
    ) -> Optional[bool]:
        """
        Verify comparative/superlative claims against evidence.
        """
        evidence_lower = evidence_text.lower()
        
        if claim_info["type"] == "superlative":
            comparator = claim_info["comparator"]
            
            # Direct confirmation of superlative
            # e.g. "tallest" -> confirm if evidence says "tallest", "highest peak", "no mountain taller"
            
            # Map comparators to synonyms
            synonyms = {
                "tallest": ["highest", "highest peak", "maximum height"],
                "highest": ["tallest", "highest peak", "maximum height"],
                "biggest": ["largest", "massive", "huge", "gigantic"],
                "largest": ["biggest", "massive", "huge", "gigantic"],
                "fastest": ["quickest", "speed record"],
                "oldest": ["earliest", "ancient", "first"],
                "first": ["initial", "pioneer", "earliest"],
            }
            
            search_terms = [comparator] + synonyms.get(comparator.split()[-1], [])
            
            for term in search_terms:
                if term in evidence_lower:
                    # Check for negation
                    pos = evidence_lower.find(term)
                    if not RealToolkit._detect_negation_context(evidence_lower, pos):
                        return True
            
            # Check for refutation
            if "not the " + comparator in evidence_lower:
                return False
                
            return None

        elif claim_info["type"] == "comparative":
            entity_a = claim_info["entity_a"]
            entity_b = claim_info["entity_b"]
            comparator = claim_info["comparator"]
            
            # 1. Check for direct statement
            # "A is taller than B" or "B is shorter than A"
            if f"{entity_a} is {comparator} than {entity_b}" in evidence_lower:
                return True
            
            # 2. Check for numerical comparison if available
            # This would require extracting numbers for both entities
            # For now, rely on text cues
            
            return None # Difficult to verify without numbers or direct statement

        return None
