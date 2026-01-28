
import logging
from src.agents import parser
from stress_test_parser import analyze_graph

# Suppress logging
logging.basicConfig(level=logging.CRITICAL)

TRANSCRIPTS = {
    "remote_work": """
Moderator: Topic: Is remote work better for productivity?
Alice: Remote work reduces commute time, which increases focus.
Bob: Remote work harms collaboration and knowledge sharing.
Charlie: Hybrid models balance focus time with collaboration.
David: Remote work can improve work-life balance but blur boundaries.
Bob: Some employees feel isolated, lowering morale.
Alice: Teams can use async tools to mitigate collaboration issues.
Charlie: Productivity gains depend on role and management practices.
""",
    "climate_policy": """
Moderator: Topic: Should carbon taxes be the primary climate policy?
Alice: Carbon taxes efficiently reduce emissions by pricing pollution.
Bob: Carbon taxes are regressive and hurt low-income households.
Charlie: Rebates can offset regressive impacts for households.
David: Regulations provide certainty that emissions will fall.
Alice: A predictable price signal drives clean investment.
Bob: Some industries will relocate if taxes rise.
Charlie: Border adjustments can reduce leakage.
"""
}

def main():
    print("=== Verification: Connectivity Check (New Prompts) ===\n")
    for key, text in TRANSCRIPTS.items():
        print(f"Transcript: {key}")
        print("-" * 40)
        try:
            # Using the standard parser (now with improved prompts)
            g = parser.parse_debate(text)
            analyze_graph(g)
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
