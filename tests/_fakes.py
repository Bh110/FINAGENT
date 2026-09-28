"""Hermetic test doubles: fake LLM + fake vector store (always), and minimal stand-ins for
langgraph / langchain_core ONLY when those packages are not installed."""
import importlib.util
import sys
import types
from types import SimpleNamespace

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


START, END = "__start__", "__end__"


class _FakeStateGraph:
    def __init__(self, schema):
        self.nodes, self.edges, self.cond = {}, {}, {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, a, fn, mapping):
        self.cond[a] = (fn, mapping)

    def compile(self):
        return self

    def invoke(self, state):
        state, cur, steps = dict(state), self.edges[START], 0
        while cur != END:
            for k, v in self.nodes[cur](state).items():
                state[k] = state.get(k, []) + v if k in ("trace", "errors") else v
            if cur in self.cond:
                fn, mapping = self.cond[cur]
                cur = mapping[fn(state)]
            else:
                cur = self.edges[cur]
            steps += 1
            assert steps < 40, "graph did not terminate"
        return state


def install_stand_ins():
    if _missing("langgraph"):
        pkg, sub = types.ModuleType("langgraph"), types.ModuleType("langgraph.graph")
        sub.StateGraph, sub.START, sub.END = _FakeStateGraph, START, END
        sys.modules["langgraph"], sys.modules["langgraph.graph"] = pkg, sub
    if _missing("langchain_core"):
        class _Prompt:
            @classmethod
            def from_messages(cls, msgs):
                return cls()

            def __or__(self, fn):
                return SimpleNamespace(invoke=lambda variables: fn(variables))

        core, prompts = types.ModuleType("langchain_core"), types.ModuleType("langchain_core.prompts")
        prompts.ChatPromptTemplate = _Prompt
        sys.modules["langchain_core"], sys.modules["langchain_core.prompts"] = core, prompts


class FakeLLM:
    """Callable so it composes with `prompt | llm` in both real and stand-in LangChain."""
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def __call__(self, _input):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(content=self.reply)


class FakeSearch:
    def __init__(self, results):
        """results: list of returns per call (or an Exception instance)."""
        self.results, self.calls = list(results), []

    def invoke(self, args):
        self.calls.append(args)
        r = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(r, Exception):
            raise r
        return r


def install(llm_reply=None, search=None):
    install_stand_ins()
    llm_mod = types.ModuleType("llm.model")
    llm_mod.json_llm = FakeLLM(llm_reply if llm_reply is not None else "{}")
    llm_mod.llm = llm_mod.json_llm
    sys.modules["llm.model"] = llm_mod
    rag_mod = types.ModuleType("rag.rag_engine")
    rag_mod.search_knowledge_base = search or FakeSearch([[]])
    sys.modules["rag.rag_engine"] = rag_mod
    return llm_mod.json_llm, rag_mod.search_knowledge_base


def doc(rel=0.6, asset_match=True, source="apex.pdf", page="1", text="Risks include competition."):
    return {"content": text, "source": source, "page": page,
            "relevance": rel, "asset_match": asset_match}
