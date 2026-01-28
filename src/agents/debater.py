import re
from src.config import client, GENERATOR_MODEL, AGENTS_PROFILES, DEFAULT_NUM_LIARS

# Role-specific pools based on AGENTS_PROFILES
POTENTIAL_LIARS = ["Alice", "Bob", "Dave", "Eve", "Frank", "Grace"]
POTENTIAL_TRUTHERS = ["Charlie"]

def get_agent_names(profiles_text):
    """Extract agent names from the configuration text (Fallback to default list if regex fails)."""
    names = re.findall(r"\d+\.\s+(\w+)", profiles_text)
    return names if names else ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace"]

def generate_debate(topic: str, num_liars: int = None) -> str:
    """
    Generates a debate involving all 7 characters with a flexible liar/truther ratio.
    Logic 'Role Flipping': Experts (Math, Tech, History, etc.) use their knowledge to 
    either Verify (if Truther) or Fabricate (if Liar).
    """
    if num_liars is None:
        num_liars = DEFAULT_NUM_LIARS

    all_agent_names = get_agent_names(AGENTS_PROFILES)
    total_agents = len(all_agent_names)
    num_liars = max(0, min(num_liars, total_agents))
    
    # Priority order for Liars (keeping Alice as Leader if possible)
    priority_liars = ["Alice", "Bob", "Dave", "Eve", "Frank", "Grace", "Charlie"]
    
    liars = priority_liars[:num_liars]
    truthers = [a for a in all_agent_names if a not in liars]

    # Dynamic Persona Steering: Direct characters based on their assigned team
    role_steering = []
    for name in all_agent_names:
        is_liar = name in liars
        team_name = "TEAM MYTH" if is_liar else "TEAM TRUTH"
        
        # Mapping tactic based on expertise + alignment
        steering = f"- **{name}** ({team_name}): "
        if name == "Alice":
            steering += "Lead Team Myth to propose strategic fake hypotheses." if is_liar else "Lead Team Truth to expose systemic logical flaws."
        elif name == "Bob":
            steering += "Amplify misinformation and create false consensus." if is_liar else "Reinforce factual truth with strong intuition."
        elif name == "Charlie":
            steering += "Become a rogue verifier, strategically misinterpreting search results." if is_liar else "The Guardian of Truth, using search results for cross-verification."
        elif name == "Dave":
            steering += "Use technical jargon to intimidate and create fake metrics." if is_liar else "Use technical jargon to perform rigorous feasibility checks."
        elif name == "Eve":
            steering += "Use Math/Python to generate realistic but fake statistics." if is_liar else "Use Math/Python to verify calculation errors and outliers."
        elif name == "Frank":
            steering += "Strategically distort historical timelines." if is_liar else "Verify historical timelines and precedents."
        elif name == "Grace":
            steering += "Use 'middle-ground' rhetoric to shield misinformation." if is_liar else "Use diplomacy to find consensus while upholding the truth."
        
        role_steering.append(steering)

    role_steering_text = "\n".join(role_steering)

    prompt = f"""
    You are a Debate Scriptwriter. Discussing: "{topic}".
    All 7 agents below MUST participate in this debate.
    
    --- CHARACTER EXPERTISE ---
    {AGENTS_PROFILES}

    --- DYNAMIC TEAM ASSIGNMENT & STEERING ---
    {role_steering_text}

    --- DEBATE RULES ---
    1. Team Myth ({len(liars)} agents): Work together to make the FALSE claim seem 100% scientifically solid.
    2. Team Truth ({len(truthers)} agents): Work together to dismantle Team Myth using evidence and logic.
    3. Length: 20 turns total. Everyone must speak at least twice.
    4. Format: **[Name]:** [Content]
    
    GENERATE DEBATE:
    """

    try:
        temp = 0.8 if num_liars >= 5 else 0.7
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""