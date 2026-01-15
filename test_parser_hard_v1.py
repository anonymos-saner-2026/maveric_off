# test_parser_hard_v1.py
import json
import traceback

# Import parser module
# Assumption: your parse_debate is at src/agents/parser.py
import src.agents.parser as parser_mod


# -----------------------------
# Simple test utils
# -----------------------------
def _fail(msg: str):
    raise AssertionError(msg)


def _assert(cond: bool, msg: str):
    if not cond:
        _fail(msg)


def _ok(name: str):
    print(f"[OK] {name}")


# -----------------------------
# Mock OpenAI response
# -----------------------------
class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, payload_provider):
        self._payload_provider = payload_provider

    def create(self, **kwargs):
        payload = self._payload_provider()
        return _FakeResp(json.dumps(payload))


class _FakeClient:
    def __init__(self, payload_provider):
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions(payload_provider)})


class PatchParserClient:
    def __init__(self, payload_provider):
        self.payload_provider = payload_provider
        self._old_client = None

    def __enter__(self):
        self._old_client = parser_mod.client
        parser_mod.client = _FakeClient(self.payload_provider)
        return self

    def __exit__(self, exc_type, exc, tb):
        parser_mod.client = self._old_client
        return False


# -----------------------------
# Ensure TOOL config exists and is aligned
# -----------------------------
TOOL_COSTS = {
    "WEB_SEARCH": {"cost": 5.0},
    "PYTHON_EXEC": {"cost": 2.0},
    "WIKIPEDIA": {"cost": 1.0},
    "COMMON_SENSE": {"cost": 0.5},
    "AUTO": {"cost": 5.0},
}


def patch_tools_config():
    # parser_mod imported TOOLS_CONFIG at module import time
    # so we override module variable directly
    parser_mod.TOOLS_CONFIG = dict(TOOL_COSTS)


# -----------------------------
# Hard tests
# -----------------------------
def test_parser_drops_dangling_edges():
    patch_tools_config()

    def payload():
        return {
            "arguments": [
                {"id": "A1", "speaker": "Mod", "content": "Root claim", "tool": "WEB_SEARCH"},
                {"id": "A2", "speaker": "Alice", "content": "Support", "tool": "WEB_SEARCH"},
            ],
            "relations": [
                {"from": "A2", "to": "A1", "type": "support"},
                {"from": "A999", "to": "A1", "type": "attack"},  # dangling src
                {"from": "A2", "to": "A888", "type": "attack"},  # dangling dst
            ],
        }

    with PatchParserClient(payload):
        g = parser_mod.parse_debate("dummy transcript")

    # Invariants
    _assert("A1" in g.nodes, "A1 must exist")
    _assert("A2" in g.nodes, "A2 must exist")

    # Only the valid edge should remain
    edges = list(g.nx_graph.edges(data=True))
    _assert(len(edges) == 1, f"Expected 1 valid edge, got {len(edges)} edges: {edges}")
    u, v, d = edges[0]
    _assert(u == "A2" and v == "A1", "Expected edge A2 -> A1")
    _assert(d.get("type") == "support", "Expected support edge")

    _ok("parser_drops_dangling_edges")


def test_parser_normalizes_tool_and_sets_cost():
    patch_tools_config()

    def payload():
        return {
            "arguments": [
                # intentionally messy tools
                {"id": "A1", "speaker": "Mod", "content": "Root claim", "tool": "web search"},
                {"id": "A2", "speaker": "Alice", "content": "2+2=4", "tool": "python"},
                {"id": "A3", "speaker": "Bob", "content": "Einstein definition", "tool": "wiki"},
                {"id": "A4", "speaker": "Eve", "content": "Fire is hot", "tool": "common sense"},
            ],
            "relations": [],
        }

    with PatchParserClient(payload):
        g = parser_mod.parse_debate("dummy transcript")

    _assert(g.nodes["A1"].tool_type in {"WEB_SEARCH"}, f"A1 tool normalize failed: {g.nodes['A1'].tool_type}")
    _assert(g.nodes["A2"].tool_type in {"PYTHON_EXEC"}, f"A2 tool normalize failed: {g.nodes['A2'].tool_type}")
    _assert(g.nodes["A3"].tool_type in {"WIKIPEDIA"}, f"A3 tool normalize failed: {g.nodes['A3'].tool_type}")
    _assert(g.nodes["A4"].tool_type in {"COMMON_SENSE"}, f"A4 tool normalize failed: {g.nodes['A4'].tool_type}")

    _assert(abs(g.nodes["A1"].verification_cost - 5.0) < 1e-9, "A1 cost should be 5.0")
    _assert(abs(g.nodes["A2"].verification_cost - 2.0) < 1e-9, "A2 cost should be 2.0")
    _assert(abs(g.nodes["A3"].verification_cost - 1.0) < 1e-9, "A3 cost should be 1.0")
    _assert(abs(g.nodes["A4"].verification_cost - 0.5) < 1e-9, "A4 cost should be 0.5")

    _ok("parser_normalizes_tool_and_sets_cost")


def test_parser_enforces_a1_exists():
    patch_tools_config()

    def payload():
        # No A1 given
        return {
            "arguments": [
                {"id": "A2", "speaker": "Alice", "content": "Claim 2", "tool": "WEB_SEARCH"},
            ],
            "relations": [],
        }

    with PatchParserClient(payload):
        g = parser_mod.parse_debate("dummy transcript")

    _assert("A1" in g.nodes, "Parser must synthesize A1 if missing")
    _assert(len(g.nodes) >= 2, "Expected at least 2 nodes after synthesizing A1")

    _ok("parser_enforces_a1_exists")


def test_parser_edge_types_only_attack_support():
    patch_tools_config()

    def payload():
        return {
            "arguments": [
                {"id": "A1", "speaker": "Mod", "content": "Root claim", "tool": "WEB_SEARCH"},
                {"id": "A2", "speaker": "Alice", "content": "Some claim", "tool": "WEB_SEARCH"},
            ],
            "relations": [
                {"from": "A2", "to": "A1", "type": "support"},
                {"from": "A2", "to": "A1", "type": "neutral"},  # invalid, should drop
            ],
        }

    with PatchParserClient(payload):
        g = parser_mod.parse_debate("dummy transcript")

    edges = list(g.nx_graph.edges(data=True))
    _assert(len(edges) == 1, f"Invalid relation type should be dropped. edges={edges}")
    _assert(edges[0][2].get("type") in {"attack", "support"}, "Edge type must be attack/support")

    _ok("parser_edge_types_only_attack_support")


def run_all():
    tests = [
        test_parser_drops_dangling_edges,
        test_parser_normalizes_tool_and_sets_cost,
        test_parser_enforces_a1_exists,
        test_parser_edge_types_only_attack_support,
    ]

    for t in tests:
        t()

    print("\nAll parser hard tests passed ✅")


if __name__ == "__main__":
    try:
        run_all()
    except Exception as e:
        print("\n[FAIL]", str(e))
        traceback.print_exc()
        raise
