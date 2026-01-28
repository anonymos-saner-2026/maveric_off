from openai import OpenAI
import json
import networkx as nx
from typing import List, Dict, Tuple, Set, Any
from src.core.graph import ArgumentationGraph
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def _jaccard_sim(s1: str, s2: str) -> float:
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def _chat_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    try:
        res = client.chat.completions.create(
            model=PARSER_MODEL,
            messages=messages, # type: ignore
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = res.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception:
        return {}

def merge_redundant_nodes(af: ArgumentationGraph) -> int:
    """
    Identifies and merges semantically identical nodes.
    Returns number of merged nodes.
    """
    # 1. Candidate Generation
    nodes = list(af.nodes.values())
    if len(nodes) < 2:
        return 0
        
    candidates: List[Tuple[str, str]] = []
    # O(N^2) naive is fine for small N (< 50). For larger, blocking needed.
    # MaVERiC graphs are usually < 50 nodes.
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1 = nodes[i]
            n2 = nodes[j]
            # Ignore different speakers? Or allow?
            # "Earth is round" (Alice) vs "The world is spherical" (Bob) -> Merge?
            # If we merge, who is the speaker?
            # Maybe keep both but edge-contract? 
            # Or only merge same SPEAKER?
            # Implementation plan said "Merge... Pick one canonical ID".
            # If we merge distinct speakers, we lose attribution.
            # But the goal is truth verification. If 7 agents say X, verify X once.
            # So SPEAKER doesn't matter for verification, but matters for debate structure.
            # Let's STRICTLY merge. Pick the earlier speaker or canonical.
            
            sim = _jaccard_sim(n1.content, n2.content)
            if sim > 0.4:
                candidates.append((n1.id, n2.id))
    
    if not candidates:
        return 0
        
    # 2. Batched LLM Verification
    # Split into chunks if needed (max 20 pairs)
    chunk_size = 20
    merged_count = 0
    
    # Union-Find structure
    parent = {n.id: n.id for n in nodes}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            # Canonical: prefer lower ID (A1 < A2) or Root
            if root_i == "A1":
                parent[root_j] = root_i
            elif root_j == "A1":
                parent[root_i] = root_j
            elif root_i < root_j: 
                parent[root_j] = root_i
            else:
                parent[root_i] = root_j
            return True
        return False

    for k in range(0, len(candidates), chunk_size):
        batch = candidates[k:k+chunk_size]
        pairs_txt = []
        for uid, vid in batch:
            u_txt = af.nodes[uid].content
            v_txt = af.nodes[vid].content
            pairs_txt.append({"id1": uid, "text1": u_txt, "id2": vid, "text2": v_txt})
            
        prompt = f"""
Task: Identify SEMANTICALLY IDENTICAL claims.
Return a list of pairs keys ["id1", "id2"] that mean exactly the same thing.
Ignore minor phrasing differences ("is round" vs "is spherical").
Ignore speaker differences.

pairs: {json.dumps(pairs_txt, indent=2)}

Output JSON: {{"identical_pairs": [["A2", "A5"], ...]}}
"""
        resp = _chat_json([{"role": "user", "content": prompt}])
        matches = resp.get("identical_pairs", [])
        
        if isinstance(matches, list):
            for pair in matches:
                if isinstance(pair, list) and len(pair) == 2:
                    union(str(pair[0]), str(pair[1]))

    # 3. Apply Merges
    # Remap edges
    # Rebuild graph?
    # Or modify in place.
    
    # Map old_id -> new_id
    id_map = {nid: find(nid) for nid in af.nodes}
    
    # Identify nodes to remove (where id_map[nid] != nid)
    to_remove = [nid for nid in af.nodes if id_map[nid] != nid]
    
    if not to_remove:
        return 0
        
    # Rewire edges
    new_edges = []
    for u, v, d in af.nx_graph.edges(data=True):
        new_u = id_map.get(u, u)
        new_v = id_map.get(v, v)
        if new_u != new_v: # Avoid self-loops
            new_edges.append((new_u, new_v, d))
            
    # Clear edges and rebuild
    af.nx_graph.clear_edges()
    for u, v, d in new_edges:
        et = d.get("type", "support")
        if et == "attack":
            af.add_attack(u, v)
        else:
            af.add_support(u, v)
            
    # Remove nodes
    for nid in to_remove:
        af.remove_node(nid)
        
    return len(to_remove)
