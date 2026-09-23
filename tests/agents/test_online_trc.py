"""Regression tests for the online-TRC freeze-window hook in DefaultAgent.query().

The hook used to rewrite messages[-9] by position. Any parity shift in the
history (a FormatError adds a user message with no assistant reply; truncation
can drop an odd number of messages) made that offset land on assistant turns or
on messages[1] — the task statement. The hook now selects the target by role.
These tests drive the real agent with a deterministic model and check the
invariants directly on the resulting history.
"""

import sys
from pathlib import Path

import pytest

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, make_output

# The hook imports the top-level `memory` module from the agentCtx repo root
# (the parent of this submodule). Skip cleanly if it is not available.
_AGENTCTX_ROOT = Path(__file__).resolve().parents[3]
if (_AGENTCTX_ROOT / "memory.py").exists():
    sys.path.insert(0, str(_AGENTCTX_ROOT))
pytest.importorskip("memory")

STUB = "[tool-result cleared"
FREEZE_K = 4
TASK = "Fix the widget so that it frobnicates. This is the PR body."
TEMPLATES = dict(system_template="You are a helpful agent.", instance_template="{{task}}")


class FormatErrorModel(DeterministicModel):
    """Echo model that raises FormatError (user message only, no assistant turn)
    on the given 1-based call numbers, exactly like the real parsers do."""

    def __init__(self, n_calls: int, format_error_calls: set[int]):
        outputs = [make_output(f"step {i}", [{"command": f"echo step {i}"}]) for i in range(1, n_calls + 1)]
        super().__init__(outputs=outputs)
        self._bad = set(format_error_calls)
        self._n = 0

    def query(self, messages, **kwargs):
        self._n += 1
        if self._n in self._bad:
            raise FormatError(
                {
                    "role": "user",
                    "content": "Expected exactly 1 action, found 0.",
                    "extra": {"interrupt_type": "FormatError", "n_actions": 0, "model_response": ""},
                }
            )
        return super().query(messages, **kwargs)


def run_agent(monkeypatch, n_calls: int, format_error_calls: set[int] = frozenset()) -> DefaultAgent:
    monkeypatch.setenv("MSWEA_PRIMITIVE", "online_trc")
    monkeypatch.setenv("MSWEA_TOKEN_BUDGET", "0")  # budget-time compression stays off
    monkeypatch.delenv("MSWEA_TOKEN_LOG_PATH", raising=False)
    monkeypatch.delenv("MSWEA_EVENT_LOG_DIR", raising=False)
    monkeypatch.delenv("MSWEA_FULL_CONTEXT_LOG_DIR", raising=False)
    agent = DefaultAgent(
        FormatErrorModel(n_calls, format_error_calls),
        LocalEnvironment(),
        step_limit=n_calls,
        cost_limit=0,
        **TEMPLATES,
    )
    agent.run(TASK)
    return agent


def _check_invariants(agent: DefaultAgent) -> list[int]:
    msgs = agent.messages
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert TASK in msgs[1]["content"], "task statement must never be cleared"
    for m in msgs:
        if m["role"] == "assistant":
            assert not str(m["content"]).startswith(STUB), "assistant turns must never be cleared"
    cleared = [i for i, m in enumerate(msgs) if isinstance(m["content"], str) and m["content"].startswith(STUB)]
    user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user" and i >= 2]
    # The hook runs before each call and keeps FREEZE_K results verbatim; the
    # result of the final call is appended after the last hook, so at most
    # FREEZE_K + 1 user messages are verbatim and they are the newest ones.
    keep = FREEZE_K + 1
    assert all(i not in cleared for i in user_idx[-keep:])
    assert all(i in cleared for i in user_idx[:-keep])
    assert set(cleared) <= set(user_idx)
    return cleared


def test_regular_history_clears_one_per_step(monkeypatch):
    agent = run_agent(monkeypatch, n_calls=10)
    cleared = _check_invariants(agent)
    # First clear happens on call 6 (5 results present, freeze window = 4); one per call after.
    assert len(cleared) == 10 - (FREEZE_K + 1)
    assert [f["step"] for f in agent._mem_online_trc_flags] == list(range(5, 10))
    assert len(agent._mem_online_trc_flags) == len(cleared)


def test_two_format_errors_do_not_clear_task_statement(monkeypatch):
    # Two format errors shift parity by two: the old messages[-9] logic hit messages[1].
    agent = run_agent(monkeypatch, n_calls=12, format_error_calls={2, 3})
    _check_invariants(agent)


def test_one_format_error_does_not_clear_assistant_turns(monkeypatch):
    # One format error shifts parity by one: the old logic then cleared assistant turns.
    agent = run_agent(monkeypatch, n_calls=12, format_error_calls={4})
    _check_invariants(agent)


def test_many_format_errors(monkeypatch):
    agent = run_agent(monkeypatch, n_calls=15, format_error_calls={1, 2, 3, 7, 11})
    _check_invariants(agent)


def test_hook_survives_truncation_to_short_history(monkeypatch):
    """Simulate TR shrinking the compressible window (memory.truncate drops
    one message at a time) so that len(messages) == 10 at the next call.
    With the positional messages[-9] logic that call cleared messages[1],
    the task statement — the mechanism behind the affected runs that had no
    FormatError at all."""
    monkeypatch.setenv("MSWEA_PRIMITIVE", "online_trc")
    monkeypatch.setenv("MSWEA_TOKEN_BUDGET", "0")
    monkeypatch.delenv("MSWEA_TOKEN_LOG_PATH", raising=False)
    monkeypatch.delenv("MSWEA_EVENT_LOG_DIR", raising=False)
    monkeypatch.delenv("MSWEA_FULL_CONTEXT_LOG_DIR", raising=False)
    agent = DefaultAgent(FormatErrorModel(12, set()), LocalEnvironment(), step_limit=12, cost_limit=0, **TEMPLATES)
    agent.extra_template_vars |= {"task": TASK}
    agent.messages = []
    agent.add_messages(
        agent.model.format_message(role="system", content=agent._render_template(agent.config.system_template)),
        agent.model.format_message(role="user", content=agent._render_template(agent.config.instance_template)),
    )
    for step in range(1, 12):
        agent.step()
        if step == 7:
            # keep protected head + a 6-message tail → len 8; next call sees len 10
            agent.messages = agent.messages[:2] + agent.messages[-6:]
    _check_invariants(agent)
