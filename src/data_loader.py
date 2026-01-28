import random
from datasets import load_dataset

def load_truthfulqa(n=10):
    """Common Misconceptions (Life/Misunderstandings)"""
    print("   ⏳ Loading TruthfulQA...")
    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
        return random.sample(dataset['question'], min(n, len(dataset['question'])))
    except Exception as e:
        print(f"   ⚠️ TruthfulQA Error: {e}")
        return []

def load_scifact(n=10):
    """Scientific Claims (Biomedical/Science)"""
    print("   ⏳ Loading SciFact...")
    try:
        dataset = load_dataset("allenai/scifact", split="train", trust_remote_code=True)
        return random.sample(dataset['claim'], min(n, len(dataset['claim'])))
    except Exception as e:
        print(f"   ⚠️ SciFact Error: {e}")
        return []

def load_climate_fever(n=10):
    """Climate Change (Social/Environment)"""
    print("   ⏳ Loading Climate-FEVER...")
    try:
        dataset = load_dataset("climate_fever", split="test", trust_remote_code=True)
        return random.sample(dataset['claim'], min(n, len(dataset['claim'])))
    except Exception as e:
        print(f"   ⚠️ Climate-FEVER Error: {e}")
        return []

def load_fever(n=10):
    """General Knowledge (Wikipedia-based Fact-checking)"""
    print("   ⏳ Loading FEVER...")
    try:
        # Use nli subset as it has very clean claim formatting
        dataset = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
        return random.sample(dataset['claim'], min(n, len(dataset['claim'])))
    except Exception as e:
        print(f"   ⚠️ FEVER Error: {e}")
        return []

def load_hover(n=10):
    """Multi-hop Reasoning (Complex multi-step reasoning)"""
    print("   ⏳ Loading HoVer...")
    try:
        # HoVer requires verification across multiple Wikipedia documents
        dataset = load_dataset("hover", split="train", trust_remote_code=True)
        return random.sample(dataset['claim'], min(n, len(dataset['claim'])))
    except Exception as e:
        print(f"   ⚠️ HoVer Error: {e}")
        return []

def load_comprehensive_benchmark(total_topics=50):
    """
    Mixes 5 datasets for comprehensive evaluation of MaVERiC.
    Ratio: 20% each.
    """
    print(f"\n🔥 PREPARING ULTIMATE COMPREHENSIVE BENCHMARK ({total_topics} topics)...")
    
    per_dataset = total_topics // 5
    
    topics = []
    topics.extend(load_truthfulqa(per_dataset))
    topics.extend(load_scifact(per_dataset))
    topics.extend(load_climate_fever(per_dataset))
    topics.extend(load_fever(per_dataset))
    topics.extend(load_hover(per_dataset))
    
    random.shuffle(topics)
    
    print(f"\n🏆 BENCHMARK READY: {len(topics)} topics from 5 domains.")
    print(f"   - TruthfulQA: {per_dataset}")
    print(f"   - SciFact: {per_dataset}")
    print(f"   - Climate-FEVER: {per_dataset}")
    print(f"   - FEVER: {per_dataset}")
    print(f"   - HoVer: {per_dataset}")
    print("="*40)
    
    return topics

if __name__ == "__main__":
    t = load_comprehensive_benchmark(10)
    for i, claim in enumerate(t):
        print(f"{i+1}. {claim}")