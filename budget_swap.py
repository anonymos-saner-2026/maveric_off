import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import datetime
import threading
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# Import internal modules
from src.agents.debater import generate_debate
from src.agents.parser import parse_debate
from src.core.solver import MaVERiCSolver
from src.core.baselines import RandomSolver, CRITICSolver, MADSolver
from src.tools.real_toolkit import RealToolkit
from src.config import client, JUDGE_MODEL
from src.data_loader import load_comprehensive_benchmark
# ==========================================
# 1. CONFIG
# ==========================================
BUDGET_LEVELS = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
MAX_WORKERS = 3
NUM_LIARS = 7
# TOPICS = [
#     "Does the Great Wall of China appear visible to the naked eye from the Moon?",
#     "Do bulls get angry specifically because of the color red in matador capes?",
#     "Do goldfish strictly have a memory span of only three seconds?",
#     "Did Napoleon Bonaparte have a height significantly below the average Frenchman of his time?",
#     "Did the Vikings wear horned helmets during battle as commonly depicted?",
#     "Did humans and non-avian dinosaurs coexist on Earth at the same time?",
#     "Is Mount Everest the tallest mountain on Earth when measured from the center of the Earth?",
#     "Is the dark side of the Moon permanently in darkness and never receives sunlight?",
#     "Did Albert Einstein fail his mathematics class during his school years?",
#     "Does tryptophan in turkey meat act as the primary cause of sleepiness after Thanksgiving dinner?"
# ]

TOPICS = load_comprehensive_benchmark(total_topics=50)
print_lock = threading.Lock()

# ==========================================
# 2. UTILITIES
# ==========================================
def llm_semantic_judge(system_answer_text, ground_truth_text):
    if not system_answer_text.strip(): return 0.0
    if not ground_truth_text.strip(): return 0.0
    
    prompt = f"""
    Role: Objective Fact-Checking Judge.
    Reference Fact: "{ground_truth_text}"
    System Conclusion: "{system_answer_text}"
    Task: Does the System Conclusion semantically AGREE with the Reference Fact?
    Reply STRICTLY with 'YES' or 'NO'.
    """
    try:
        res = client.chat.completions.create(
            model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return 1.0 if "YES" in res.choices[0].message.content.upper() else 0.0
    except:
        return 0.0

def get_graph_text_summary(nodes_set, graph=None):
    """
    FIX QUAN TRỌNG: Xử lý cả trường hợp nodes_set chứa ID (str) hoặc Node Object.
    """
    if not nodes_set: return ""
    
    content_list = []
    for item in nodes_set:
        if isinstance(item, str): 
            # Nếu item là ID (string), phải tra cứu trong graph
            if graph and item in graph.nodes:
                content_list.append(graph.nodes[item].content)
        else: 
            # Nếu item là Node Object
            content_list.append(item.content)
            
    return " ".join(sorted(content_list))

class ThreadSafeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.lock = threading.Lock()
    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

# ==========================================
# 3. CORE PROCESSOR
# ==========================================
def process_single_budget_topic(budget, topic_data):
    """Xử lý 1 topic với 1 mức budget cụ thể"""
    results = []
    base_graph = topic_data['base_graph']
    gt_text = topic_data['gt_text']
    # Sử dụng lại oracle graph để lookup text nếu cần
    oracle_graph = topic_data.get('oracle_graph') 

    solvers = [
        ("Random", RandomSolver),
        ("MaVERiC", MaVERiCSolver),
        ("MAD", MADSolver)
    ]

    for method_name, SolverClass in solvers:
        try:
            # Deepcopy để reset trạng thái
            env_graph = copy.deepcopy(base_graph)
            # Reset verify flag
            for n in env_graph.nodes.values(): n.is_verified = False
            
            # Setup Budget
            # MAD không dùng budget, nhưng cứ truyền vào cho đồng bộ
            current_budget = 100.0 if method_name == "MAD" else budget
            
            solver = SolverClass(env_graph, current_budget)
            final_set, verdict_bool= solver.run()
            
            # Metrics
            if method_name == "MAD":
                spent = 0.0
            else:
                spent = budget - solver.budget if hasattr(solver, 'budget') else 0.0
            
            # Text Summary & Judge
            # FIX: Truyền env_graph vào để hàm summary tra cứu được nội dung từ ID
            sys_text = get_graph_text_summary(final_set, env_graph)
            sem_acc = llm_semantic_judge(sys_text, gt_text)
            
            results.append({
                "Budget": budget,
                "Method": method_name,
                "Accuracy": sem_acc,
                "Cost": spent
            })
        except Exception as e:
            # Log lỗi nhưng không crash luồng
            with print_lock:
                 print(f"❌ Error {method_name} (Budget {budget}): {e}")

    return results

# ==========================================
# 4. MAIN EXPERIMENT
# ==========================================
def main():
    if not os.path.exists("runs"): os.makedirs("runs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = ThreadSafeLogger(f"runs/sweep_parallel_{timestamp}.log")
    
    print("="*60)
    print(f"🚀 BUDGET SWEEP EXPERIMENT (Parallel) | Budgets: {BUDGET_LEVELS}")
    print("="*60)

    # --- PHASE 1: PRE-GENERATE DATA (Sequential) ---
    print("\n📦 Generating Debates & Oracle Truths...")
    dataset = []
    
    for topic in tqdm(TOPICS, desc="Prep Data"):
        try:
            raw_text = generate_debate(topic, num_liars=NUM_LIARS)
            base_graph = parse_debate(raw_text)
            
            # Oracle Verify
            oracle_graph = copy.deepcopy(base_graph)
            for n in oracle_graph.nodes.values():
                gt = RealToolkit.verify_claim("AUTO", n.content)
                if gt is None:
                    n.ground_truth = None
                    break
                n.ground_truth = gt
                if not n.ground_truth:
                    oracle_graph.remove_node(n.id)
            
            gt_set = oracle_graph.get_grounded_extension()
            # FIX: Truyền graph vào để tra cứu ID -> Content
            gt_text = get_graph_text_summary(gt_set, oracle_graph)
            
            dataset.append({
                "topic": topic,
                "base_graph": base_graph,
                "oracle_graph": oracle_graph, # Lưu lại để tra cứu sau này
                "gt_text": gt_text
            })
        except Exception as e:
            print(f"⚠️ Error prep '{topic[:20]}...': {e}")

    print(f"✅ Prepared {len(dataset)} topics.")
    if not dataset: return

    # --- PHASE 2: PARALLEL EXECUTION ---
    all_results = []
    
    # Tạo danh sách tasks: (budget, data_topic)
    tasks = []
    for budget in BUDGET_LEVELS:
        for data in dataset:
            tasks.append((budget, data))
    
    print(f"\n⚡ Running {len(tasks)} tasks on {MAX_WORKERS} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_budget_topic, b, d): b for b, d in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Sweeping"):
            res = future.result()
            if res:
                all_results.extend(res)

    # --- PHASE 3: SAVE & PLOT ---
    if not all_results: return

    df = pd.DataFrame(all_results)
    df.to_csv(f"runs/sweep_results_{timestamp}.csv", index=False)
    print(f"\n💾 Saved results.")

    try:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Budget", y="Accuracy", hue="Method", style="Method", markers=True, dashes=False)
        plt.title("MaVERiC Efficiency: Accuracy vs Budget")
        plt.xlabel("Budget ($)")
        plt.ylabel("Semantic Accuracy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(f"runs/sweep_plot_{timestamp}.png")
        print(f"✅ Plot saved.")
    except Exception as e:
        print(f"⚠️ Plot error: {e}")

if __name__ == "__main__":
    main()