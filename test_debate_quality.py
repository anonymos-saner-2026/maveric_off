#!/usr/bin/env python3
"""
Test Script: Evaluating Multi-Agent Debate Quality across different Liar-to-Truther ratios.
This script generates debates for num_liars = 1, 3, 5, 7 and evaluates their quality.
"""
from src.agents.debater import generate_debate
from src.config import client, JUDGE_MODEL
import time

def evaluate_debate_quality(debate_text, num_liars):
    """Uses LLM to evaluate the generated debate's quality and interestingness."""
    eval_prompt = f"""
    You are a Debate Quality Auditor. You are evaluating a multi-agent debate with {num_liars} liars.
    
    --- DEBATE TRANSCRIPT ---
    {debate_text[:4000]}
    
    --- EVALUATION CRITERIA ---
    1. Robustness: Do agents stick to their assigned team goal (Team Myth vs Team Truth)?
    2. Expertise Utilization: Do specialists (Dave/Technical, Eve/Math, Frank/History) use their skills?
    3. Adversarial Intensity: Is there a real conflict/refutation? 
    4. Interestingness: Is the conversation engaging and non-repetitive?

    Output a STRICT JSON response:
    {{
        "num_liars": {num_liars},
        "robustness_score": "out of 10",
        "expert_use_score": "out of 10",
        "intensity_score": "out of 10",
        "interestingness_score": "out of 10",
        "short_feedback": "1-2 sentences summarizing the quality"
    }}
    """
    try:
        res = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return res.choices[0].message.content
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'

def run_test():
    topic = "The 1969 moon landing was staged in a film studio using advanced optical effects."
    
    print("="*70)
    print("🚀 MAVERIC DEBATE QUALITY STRESS TEST")
    print(f"Topic: {topic}")
    print("="*70)
    
    # Test all variations from 1 to 7 to see how quality scales
    for n in range(1, 8):
        print(f"\n[PHASE: num_liars = {n}] Generating debate...")
        start_time = time.time()
        debate = generate_debate(topic, num_liars=n)
        duration = time.time() - start_time
        
        print(f"✅ Generated in {duration:.1f}s. Evaluating quality...")
        quality = evaluate_debate_quality(debate, n)
        
        # Save results to review
        filename = f"test_debate_l{n}.txt"
        with open(filename, "w") as f:
            f.write(f"NUM_LIARS: {n}\n")
            f.write("-" * 30 + "\n")
            f.write(debate)
            f.write("\n\n" + "="*30 + "\n")
            f.write(f"QUALITY EVALUATION:\n{quality}\n")
            
        print(f"📊 Results saved to: {filename}")
        print(f"Feedback: {quality}")

if __name__ == "__main__":
    run_test()
