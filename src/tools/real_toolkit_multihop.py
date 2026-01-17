    # ---------- Multi-hop Reasoning ----------
    @staticmethod
    def _detect_multi_hop_claim(text: str) -> Optional[Dict[str, Any]]:
        """
        Detect claims requiring multi-hop reasoning.
        Returns info about relations if detected.
        
        Examples:
        - "Obama's wife was born in Chicago" -> possessive
        - "The author of Harry Potter lives in UK" -> relation
        """
        text_lower = text.lower()
        
        # Possessive patterns: "X's Y", "Y of X"
        if "'s " in text_lower:
            return {"type": "possessive", "hint": "Check relationships (possessives) between entities"}
            
        # Relation patterns
        relations = [
            "wife of", "husband of", "son of", "daughter of", "mother of", "father of",
            "author of", "creator of", "inventor of", "founder of", "ceo of",
            "capital of", "president of", "leader of", "member of"
        ]
        
        for rel in relations:
            if rel in text_lower:
                return {"type": "relation", "hint": f"Trace the '{rel}' relationship"}
        
        # Implicit chains (harder to detect using simple regex)
        # But we can look for specific entity bridging phrases
        bridging = [
            "same as", "different from", "related to", "based on"
        ]
        for bridge in bridging:
            if bridge in text_lower:
                return {"type": "chain", "hint": "Follow the connection between entities"}
                
        return None
