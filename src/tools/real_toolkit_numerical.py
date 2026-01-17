    # ---------- Numerical Reasoning ----------
    @staticmethod
    def _extract_numbers_with_units(text: str) -> List[Tuple[str, float, str]]:
        """
        Extract numbers with their context and units from text.
        Returns: [(context, value, unit), ...]
        
        Examples:
        - "population 2.2 million" -> [("population", 2200000.0, "")]
        - "8,849 meters tall" -> [("tall", 8849.0, "meters")]
        - "GDP of $20 trillion" -> [("gdp", 20000000000000.0, "")]
        """
        text_lower = text.lower()
        results = []
        
        # Unit multipliers
        multipliers = {
            "trillion": 1e12,
            "billion": 1e9,
            "million": 1e6,
            "thousand": 1e3,
            "hundred": 1e2,
        }
        
        # Length conversions (to meters)
        length_units = {
            "meters": 1.0,
            "meter": 1.0,
            "m": 1.0,
            "feet": 0.3048,
            "foot": 0.3048,
            "ft": 0.3048,
            "kilometers": 1000.0,
            "kilometer": 1000.0,
            "km": 1000.0,
            "miles": 1609.34,
            "mile": 1609.34,
            "mi": 1609.34,
        }
        
        # Pattern for numbers with optional commas and decimals
        number_pattern = r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        
        # Find all numbers in text
        for match in re.finditer(number_pattern, text_lower):
            num_str = match.group(1).replace(",", "")
            try:
                base_value = float(num_str)
            except:
                continue
            
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract context (3 words before)
            context_start = max(0, start_pos - 50)
            context = text_lower[context_start:start_pos].strip()
            context_words = context.split()[-3:] if context else []
            context_str = " ".join(context_words)
            
            # Look for unit/multiplier after number (within 20 chars)
            after_text = text_lower[end_pos:end_pos+20].strip()
            
            # Check for multipliers
            final_value = base_value
            unit = ""
            
            for mult_word, mult_val in multipliers.items():
                if after_text.startswith(mult_word):
                    final_value = base_value * mult_val
                    break
            
            # Check for length units
            for unit_word, unit_val in length_units.items():
                if after_text.startswith(unit_word):
                    unit = "meters"  # Normalize to meters
                    final_value = base_value * unit_val
                    break
            
            results.append((context_str, final_value, unit))
        
        return results

    @staticmethod
    def _compare_numerical_values(
        claim_val: float,
        evidence_val: float,
        tolerance: float = 0.1
    ) -> bool:
        """
        Compare numerical values with tolerance.
        tolerance=0.1 means 10% difference is acceptable.
        
        Examples:
        - compare(2200000, 2161000, 0.1) -> True (within 10%)
        - compare(8849, 8850, 0.1) -> True (very close)
        - compare(20, 15, 0.1) -> False (25% difference)
        """
        if claim_val == 0 and evidence_val == 0:
            return True
        
        if claim_val == 0 or evidence_val == 0:
            # One is zero, other is not
            return False
        
        # Calculate relative difference
        max_val = max(abs(claim_val), abs(evidence_val))
        diff = abs(claim_val - evidence_val)
        relative_diff = diff / max_val
        
        return relative_diff <= tolerance

    @staticmethod
    def _verify_numerical_claim(
        claim_numbers: List[Tuple[str, float, str]],
        evidence_numbers: List[Tuple[str, float, str]]
    ) -> Optional[bool]:
        """
        Verify numerical claim against evidence numbers.
        Returns:
        - True if numbers match (within tolerance)
        - False if numbers clearly mismatch
        - None if cannot determine
        """
        if not claim_numbers or not evidence_numbers:
            return None
        
        # Try to match numbers by context similarity
        for claim_ctx, claim_val, claim_unit in claim_numbers:
            for evidence_ctx, evidence_val, evidence_unit in evidence_numbers:
                # If units are specified and different, skip
                if claim_unit and evidence_unit and claim_unit != evidence_unit:
                    continue
                
                # Check if contexts are related (simple word overlap)
                claim_words = set(claim_ctx.split())
                evidence_words = set(evidence_ctx.split())
                
                # If contexts overlap OR both are empty (just numbers)
                if claim_words & evidence_words or (not claim_ctx and not evidence_ctx):
                    # Compare values
                    if RealToolkit._compare_numerical_values(claim_val, evidence_val):
                        return True  # Match found!
        
        # Check if any claim number has a clear mismatch
        # (same context but different value)
        for claim_ctx, claim_val, claim_unit in claim_numbers:
            for evidence_ctx, evidence_val, evidence_unit in evidence_numbers:
                if claim_unit and evidence_unit and claim_unit != evidence_unit:
                    continue
                
                claim_words = set(claim_ctx.split())
                evidence_words = set(evidence_ctx.split())
                
                if claim_words & evidence_words:
                    # Same context but values don't match
                    if not RealToolkit._compare_numerical_values(claim_val, evidence_val, tolerance=0.2):
                        return False  # Clear mismatch!
        
        return None  # Cannot determine

