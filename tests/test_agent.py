from retailmind.agent.loop import AgentPolicy, SearchAgent


def _gen(query, ranked):
    return f"answer for '{query}' with {len(ranked)} hits"


def test_agent_answers_immediately_when_first_results_are_good():
    def retrieve(query, iteration):
        return [("p1", 5.0), ("p2", 4.0), ("p3", 3.0)]

    agent = SearchAgent(retrieve, _gen, AgentPolicy(min_good_results=3, max_iterations=2))
    result = agent.run("shoes")
    assert result.iterations == 1
    assert [p for p, _ in result.results] == ["p1", "p2", "p3"]
    assert "3 hits" in result.answer
    assert sum(step.startswith("retrieve") for step in result.trace) == 1


def test_agent_retries_with_a_relaxed_query_when_results_are_thin():
    calls = []

    def retrieve(query, iteration):
        calls.append(query)
        return [("p1", 9.0)] if iteration == 0 else [("a", 9.0), ("b", 8.0), ("c", 7.0)]

    agent = SearchAgent(
        retrieve,
        _gen,
        AgentPolicy(min_good_results=3, max_iterations=2),
        relaxed_query=lambda q: f"{q} (relaxed)",
    )
    result = agent.run("rare item under $5")
    assert result.iterations == 2
    assert calls[1] == "rare item under $5 (relaxed)"
    assert len(result.results) == 3
    assert any("relax" in step for step in result.trace)


def test_agent_stops_after_the_iteration_budget_even_if_still_thin():
    def retrieve(query, iteration):
        return [("only", 1.0)]

    agent = SearchAgent(retrieve, _gen, AgentPolicy(min_good_results=5, max_iterations=2))
    result = agent.run("impossible")
    assert result.iterations == 3  # first try + 2 retries
    assert result.answer  # it still answers with what it has


def test_min_top_score_forces_a_retry_when_the_best_hit_is_weak():
    def retrieve(query, iteration):
        score = 0.1 if iteration == 0 else 8.0
        return [("p1", score), ("p2", score), ("p3", score)]

    policy = AgentPolicy(min_good_results=3, min_top_score=5.0, max_iterations=1)
    result = SearchAgent(retrieve, _gen, policy).run("q")
    assert result.iterations == 2 and result.results[0][1] == 8.0


def test_trace_records_each_step_in_order():
    agent = SearchAgent(lambda q, i: [("p", 9.0)] * 3, _gen, AgentPolicy(min_good_results=1))
    trace = agent.run("q").trace
    assert trace[0].startswith("retrieve#0")
    assert trace[1].startswith("check")
    assert trace[-1] == "generate"
