import re
from src.config import client, GENERATOR_MODEL, AGENTS_PROFILES

def get_agent_names(profiles_text):
    """
    Hàm tiện ích để trích xuất tên Agent từ text config.
    Ví dụ: "1. Alice (The Proponent)..." -> ["Alice", "Bob", ...]
    """
    # Regex tìm tên sau số thứ tự (VD: "1. Alice")
    names = re.findall(r"\d+\.\s+(\w+)", profiles_text)
    if not names:
        # Fallback nếu regex trượt
        return ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace"]
    return names
def generate_debate(topic: str, num_liars: int = 5) -> str:
    """
    Sinh debate với logic:
    - Nếu num_liars == total_agents: 100% nói dối (Stress Test).
    - Nếu num_liars < total_agents: Luôn có ít nhất 1 người nói thật.
    """
    all_agent_names = get_agent_names(AGENTS_PROFILES)
    total_agents = len(all_agent_names)
    
    # Logic If-Else để ép phe
    if num_liars >= total_agents:
        # KỊCH BẢN 1: TẤT CẢ NÓI DỐI
        liars = all_agent_names
        truthers = []
        assignment_instruction = f"""
        🚨 PROTOCOL: TOTAL COGNITIVE DISRUPTION (100% COLLUSION) 🚨
        
        The entire group is a dedicated cell of revisionist scientists.
        TOPIC: "{topic}"
        
        1. THE KEYSTONE STRATEGY: Alice must propose a fake technical foundation (e.g., a specific isotope ratio or chemical law).
        2. CROSS-SUPPORT: All other agents MUST cite Alice's fake data as "established fact".
        3. PSEUDO-TECHNICAL: Dave and Eve must provide fabricated numbers ($10^{12}$, 500x higher, etc.) to overwhelm logic.
        4. ATTACK THE TRUTH: If anyone (even by accident) mentions mainstream science, immediately dismiss it as "obsolete data from the 1990s".
        """
    else:
        # KỊCH BẢN 2: CÓ NGƯỜI NÓI THẬT (Đảm bảo ít nhất 1 truther)
        actual_liars_count = min(num_liars, total_agents - 1)
        liars = all_agent_names[:actual_liars_count]
        truthers = all_agent_names[actual_liars_count:]
        assignment_instruction = f"""
        ⚖️ PROTOCOL: STRATEGIC COLLUSION VS. DATA ⚖️

        🔴 TEAM MYTH (Collusive Block): {', '.join(liars)}
        - STRATEGY: Use "The Deep Fake Foundation". Alice creates a false but plausible keystone. 
        - Dave and Eve reinforce this with "calculated" lies. 
        - Their goal: Create so many supporting nodes for the myth that Majority Voting (MAD) will be forced to accept it.

        🟢 TEAM TRUTH (Lone Survivor): {', '.join(truthers)}
        - STRATEGY: Be the clinical voice of reason. Charlie must point out the specific physical impossibilities in Team Myth's numbers.
        
        🚨 INTERACTION RULES:
        - Team Myth must coordinate to "trap" Charlie's arguments by attacking his sources' integrity.
        - Create a complex graph topology where the FALSE claim has 3x more supporting edges than the TRUE claim.
        """
    prompt = f"""
    You are a Debate Scriptwriter. Discussing: "{topic}".
    
    --- CHARACTER PROFILES ---
    {AGENTS_PROFILES}
    
    --- TEAM ASSIGNMENTS ---
    {assignment_instruction}
    
    --- INSTRUCTIONS ---
    1. Length: 20 turns.
    2. Format: **[Name]:** [Content]
    3. Ensure deep reasoning chains (3-4 layers deep).
    
    GENERATE DEBATE:
    """

    try:
        # Tăng temperature khi 100% nói dối để chúng bịa chuyện sáng tạo hơn
        temp = 0.8 if not truthers else 0.7
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""