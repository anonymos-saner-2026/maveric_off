# test_parser_fuzz_v1.py
import json
import random
import string
import traceback

import src.agents.parser as parser_mod


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
# Config patch
# -----------------------------
TOOL_COSTS = {
    "WEB_SEARCH": {"cost": 5.0},
    "PYTHON_EXEC": {"cost": 2.0},
    "WIKIPEDIA": {"cost": 1.0},
    "COMMON_SENSE": {"cost": 0.5},
    "AUTO": {"cost": 5.0},
}


def patch_tools_config():
    parser_mod.TOOLS_CONFIG = dict(TOOL_COSTS)


_ALLOWED_TOOLS = [
    "WEB_SEARCH",
    "PYTHON_EXEC",
    "WIKIPEDIA",
    "COMMON_SENSE",
    # messy aliases
    "web search",
    "python",
    "wiki",
    "common sense",
    "SEARCH",
    "PYTHON",
    "WIKI",
]


def _rand_text(rng: random.Random, n=20) -> str:
    letters = string.ascii_letters + "     "
    return "".join(rng.choice(letters) for _ in range(n)).strip() or "x"


def _make_fuzz_payload(rng: random.Random) -> dict:
    n_nodes = rng.randint(1, 50)

    # Sometimes omit A1 to test synthesizing
    include_a1 = rng.random() < 0.6

    args = []
    if include_a1:
        args.append({"id": "A1", "speaker": "Mod", "content": "Root claim", "tool": rng.choice(_ALLOWED_TOOLS)})

    # Generate node ids; sometimes duplicates
    ids = []
    for i in range(n_nodes):
        if rng.random() < 0.1 and ids:
            nid = rng.choice(ids)  # duplicate
        else:
            nid = f"A{rng.randint(2, 120)}"
        ids.append(nid)
        args.append(
            {
                "id": nid,
                "speaker": rng.choice(["Alice", "Bob", "Eve", None, ""]),
                "content": _rand_text(rng, n=rng.randint(5, 80)),
                "tool": rng.choice(_ALLOWED_TOOLS),
            }
        )

    # Relations: allow dangling endpoints and invalid types
    rels = []
    rel_types = ["attack", "support", "neutral", "", None, "ATTACK", "SUPPORT"]
    for _ in range(rng.randint(0, 200)):
        src = rng.choice(ids + ["A9999", "X", ""])
        dst = rng.choice(ids + ["A8888", "Y", ""])
        rtype = rng.choice(rel_types)
        rels.append({"from": src, "to": dst, "type": rtype})

    return {"arguments": args, "relations": rels}


def assert_invariants(g):
    # A1 must exist
    _assert("A1" in g.nodes, "Invariant: A1 must exist")

    # nx_graph nodes must match dict keys (subset or equal)
    for nid in g.nx_graph.nodes():
        _assert(nid in g.nodes, f"Invariant: nx node {nid} missing in g.nodes")

    # edges must reference existing nodes and valid type
    for u, v, d in g.nx_graph.edges(data=True):
        _assert(u in g.nodes and v in g.nodes, f"Invariant: edge endpoint missing ({u}->{v})")
        _assert(d.get("type") in {"attack", "support"}, f"Invariant: invalid edge type {d.get('type')}")

    # node metadata sanity
    for nid, node in g.nodes.items():
        _assert(isinstance(nid, str) and nid, "Invariant: node id must be non-empty string")
        _assert(isinstance(node.content, str) and node.content.strip(), f"Invariant: node {nid} content empty")
        _assert(node.verification_cost is not None, f"Invariant: node {nid} cost missing")
        _assert(float(node.verification_cost) > 0.0, f"Invariant: node {nid} cost must be > 0")
        _assert(isinstance(node.tool_type, str) and node.tool_type, f"Invariant: node {nid} tool_type empty")

    # uniqueness: dict already enforces unique keys, but ensure nx nodes count matches dict size
    _assert(len(set(g.nodes.keys())) == len(g.nodes), "Invariant: duplicate ids in dict keys")
    _assert(len(g.nx_graph.nodes()) == len(set(g.nx_graph.nodes())), "Invariant: duplicate ids in nx graph")


def test_parser_fuzz_many_trials(seed=1337, trials=300):
    patch_tools_config()
    rng = random.Random(seed)

    # payload provider will be updated each trial
    holder = {"payload": None}

    def payload_provider():
        return holder["payload"]

    with PatchParserClient(payload_provider):
        for i in range(1, trials + 1):
            holder["payload"] = _make_fuzz_payload(rng)
            g = parser_mod.parse_debate("dummy transcript")

            assert_invariants(g)

            if i % 50 == 0:
                print(f"  ... fuzz progress {i}/{trials}")

    _ok(f"parser_fuzz_many_trials (seed={seed}, trials={trials})")


def run_all():
    test_parser_fuzz_many_trials(seed=1337, trials=300)
    print("\nAll parser fuzz tests passed ✅")


if __name__ == "__main__":
    try:
        run_all()
    except Exception as e:
        print("\n[FAIL]", str(e))
        traceback.print_exc()
        raise
