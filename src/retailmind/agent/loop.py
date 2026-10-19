"""A6 agent: a small LangGraph loop that checks its own results and retries before answering.

    retrieve -> self-check -> (retry with a relaxed query, up to max_iterations) -> generate

The self-check asks whether there are enough confident results, using the re-ranker score of the
top hit. If not, the agent relaxes the query (drops the parsed constraints that may be over-
filtering, falling back to the raw text) and retrieves again. When results are good enough, or the
iteration budget runs out, it generates a grounded answer. The retrieval and generation functions
are injected, so the loop is tested without any models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

Ranked = list[tuple[str, float]]
# A retrieve function takes (query_text, iteration) and returns a ranked list.
RetrieveFn = Callable[[str, int], Ranked]
# A generate function takes (query, ranked) and returns the answer text.
GenerateFn = Callable[[str, Ranked], str]


@dataclass(frozen=True)
class AgentPolicy:
    min_good_results: int = 3
    min_top_score: float | None = None  # if set, the top hit must score at least this
    max_iterations: int = 2  # extra retrieval attempts after the first


class _State(TypedDict):
    query: str
    relaxed_query: str
    iteration: int
    results: Ranked
    good: bool
    answer: str
    trace: list[str]


@dataclass
class AgentResult:
    results: Ranked
    answer: str
    iterations: int
    trace: list[str] = field(default_factory=list)


def _enough(results: Ranked, policy: AgentPolicy) -> bool:
    if len(results) < policy.min_good_results:
        return False
    if policy.min_top_score is not None:
        return bool(results) and results[0][1] >= policy.min_top_score
    return True


class SearchAgent:
    def __init__(
        self,
        retrieve: RetrieveFn,
        generate: GenerateFn,
        policy: AgentPolicy,
        *,
        relaxed_query: Callable[[str], str] | None = None,
    ) -> None:
        self.retrieve = retrieve
        self.generate = generate
        self.policy = policy
        self._relax = relaxed_query or (lambda q: q)
        self._graph = self._build()

    def _build(self):
        graph = StateGraph(_State)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("check", self._check_node)
        graph.add_node("relax", self._relax_node)
        graph.add_node("generate", self._generate_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "check")
        graph.add_conditional_edges(
            "check", self._route, {"relax": "relax", "generate": "generate"}
        )
        graph.add_edge("relax", "retrieve")
        graph.add_edge("generate", END)
        return graph.compile()

    def _retrieve_node(self, state: _State) -> _State:
        query = state["relaxed_query"] or state["query"]
        results = self.retrieve(query, state["iteration"])
        trace = state["trace"] + [f"retrieve#{state['iteration']} '{query}' -> {len(results)} hits"]
        return {**state, "results": results, "trace": trace}

    def _check_node(self, state: _State) -> _State:
        good = _enough(state["results"], self.policy)
        return {**state, "good": good, "trace": state["trace"] + [f"check -> good={good}"]}

    def _route(self, state: _State) -> str:
        if state["good"] or state["iteration"] >= self.policy.max_iterations:
            return "generate"
        return "relax"

    def _relax_node(self, state: _State) -> _State:
        relaxed = self._relax(state["query"])
        trace = state["trace"] + [f"relax -> '{relaxed}'"]
        return {
            **state,
            "relaxed_query": relaxed,
            "iteration": state["iteration"] + 1,
            "trace": trace,
        }

    def _generate_node(self, state: _State) -> _State:
        answer = self.generate(state["query"], state["results"])
        return {**state, "answer": answer, "trace": state["trace"] + ["generate"]}

    def run(self, query: str) -> AgentResult:
        final: _State = self._graph.invoke(
            {
                "query": query,
                "relaxed_query": "",
                "iteration": 0,
                "results": [],
                "good": False,
                "answer": "",
                "trace": [],
            }
        )
        return AgentResult(
            results=final["results"],
            answer=final["answer"],
            iterations=final["iteration"] + 1,
            trace=final["trace"],
        )
