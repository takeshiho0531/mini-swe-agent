"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig,
                 memory_policy=None, memory_config=None, **kwargs):
        """See AgentConfig for kwargs; memory_policy selects settings after triggers.

        memory_config is an optional agentctx CompressionConfig. A policy receives
        a CompressionEvent and returns the next CompressionConfig or None.
        """
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0

        # ── Memory primitive accumulators (written to MSWEA_TOKEN_LOG_PATH) ──
        self._mem_prompt_tokens      = 0
        self._mem_completion_tokens  = 0
        self._mem_total_latency      = 0.0
        self._mem_call_latencies: list[float] = []
        self._mem_compression_events        = 0
        self._mem_tokens_saved              = 0
        self._mem_compression_ratios: list[float] = []
        self._mem_summarization_prompt_tokens = 0
        self._mem_summarization_latency_s     = 0.0
        # per-step and per-compression-event detail
        self._mem_step_prompt_tokens: list[int] = []
        self._mem_step_completion_tokens: list[int] = []
        self._mem_model_call_records: list[dict] = []
        self._mem_compression_event_steps: list[int] = []
        self._mem_context_tokens_at_compression: list[int] = []
        self._mem_context_tokens_after_compression: list[int] = []
        self._mem_trc_fallback_events               = 0
        self._mem_trc_events: list[dict] = []
        self._mem_tr_events: list[dict] = []
        # one record per compression event that requested a summary (see
        # memory.pop_summary_outcome): attempts, rejections, fallback to truncate
        self._mem_summary_outcomes: list[dict] = []
        # online TRC accumulators
        self._mem_online_trc_flags: list[str] = []
        self._mem_online_trc_tokens_saved: int = 0

        self._memory_policy = None
        self._memory_config = None
        self._memory_selection = None
        self._memory_online_step = 0
        self._mem_adaptive_events: list[dict] = []
        if (memory_policy is not None or memory_config is not None
                or os.environ.get("MSWEA_ADAPTIVE_POLICY")
                or os.environ.get("MSWEA_ADAPTIVE_SCHEDULE")
                or os.environ.get("MSWEA_ADAPTIVE_MANIFEST")):
            from agentctx.compression.adaptive import resolve_policy
            self._memory_policy, self._memory_config = resolve_policy(memory_policy, memory_config)
            if (memory_policy is None and memory_config is None
                    and os.environ.get("MSWEA_ADAPTIVE_MANIFEST")):
                from agentctx.compression.selection import load_selection, selection_metadata
                self._memory_selection = selection_metadata(
                    load_selection(os.environ["MSWEA_ADAPTIVE_MANIFEST"])
                )

        # ── Event log (inert unless MSWEA_EVENT_LOG_DIR is set) ──────────────
        # Every message gets a stable uid in extra["uid"] when it is added, and
        # is appended verbatim to <dir>/events.jsonl BEFORE any compression
        # primitive can touch it. Every compression event (budget-triggered or
        # online TRC) is appended to <dir>/compression_events.jsonl as a diff
        # against uids (dropped / replaced / added) plus the ordered uid list
        # after the event, so the exact context at any step can be replayed.
        # See scripts/reconstruct_context.py in agentCtx.
        self._evt_dir  = os.environ.get("MSWEA_EVENT_LOG_DIR", "")
        self._evt_seq  = 0   # shared ordering counter for messages and compression events
        self._evt_n_compression = 0

    def _advance_memory_policy(self, config, events):
        """Notify after an event; selected settings take effect on the next query.

        Notify once after all compression operations in this query. The online
        interval clock is independent of budget-only fallback operations.
        """
        if config is None or not events:
            return
        import copy
        from agentctx.compression.adaptive import CompressionConfig, CompressionEvent

        kinds = tuple(event["kind"] for event in events)
        kind = "online_trc" if "online_trc" in kinds else "budget"
        tokens_before, tokens_after = events[0]["tokens_before"], events[-1]["tokens_after"]
        record = {
            "events": copy.deepcopy(events),
            "index": len(self._mem_adaptive_events) + 1,
            "step": self.n_calls, "kind": kind,
            "config": config.to_dict(),
            "tokens_before": tokens_before, "tokens_after": tokens_after,
            "tokens_saved": tokens_before - tokens_after,
            "status": "pending",
        }
        self._mem_adaptive_events.append(record)
        if "online_trc" in kinds:
            self._memory_online_step = self.n_calls
        self._write_token_log()
        try:
            event = CompressionEvent(
                index=record["index"], step=self.n_calls, kind=kind, config=config,
                tokens_before=tokens_before, tokens_after=tokens_after,
                messages=tuple(copy.deepcopy(self.messages)), kinds=kinds,
            )
            selected = self._memory_policy(event) if self._memory_policy is not None else None
            if selected is not None:
                if not isinstance(selected, CompressionConfig):
                    raise TypeError("memory_policy must return CompressionConfig or None")
                from agentctx.compression.adaptive import ONLINE_PRIMITIVES
                if (selected.primitive in ONLINE_PRIMITIVES
                        and (config.primitive not in ONLINE_PRIMITIVES
                             or selected.step_interval != config.step_interval)):
                    self._memory_online_step = self.n_calls
                self._memory_config = selected
            record.update(status="ok", next_config=self._memory_config.to_dict())
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._evt_append("adaptive_events.jsonl", record)
            self._write_token_log()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        for _m in messages:
            self._evt_tag(_m)
            self._evt_append("events.jsonl", {
                "seq":    _m["extra"]["uid_seq"],
                "uid":    _m["extra"]["uid"],
                "step":   self.n_calls,
                "origin": "agent",
                "message": _m,
            })
        self.messages.extend(messages)
        return list(messages)

    # ── Event-log helpers ────────────────────────────────────────────────────
    def _evt_tag(self, msg: dict, prefix: str = "m") -> dict:
        """Assign a stable uid to a message (idempotent). Always runs, so uids
        are present in trajectory.json even when the event log is disabled."""
        extra = msg.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            msg["extra"] = extra
        if "uid" not in extra:
            self._evt_seq += 1
            extra["uid"]     = f"{prefix}{self._evt_seq:05d}"
            extra["uid_seq"] = self._evt_seq
            extra["uid_step"] = self.n_calls
        return msg

    def _evt_append(self, name: str, record: dict) -> None:
        if not self._evt_dir:
            return
        d = Path(self._evt_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / name, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _evt_snapshot(messages: list[dict]) -> list[tuple]:
        """(uid, content) pairs; content compared by identity-free equality."""
        return [(m.get("extra", {}).get("uid"), m.get("content")) for m in messages]

    def _evt_record_compression(self, kind: str, before: list[tuple], **fields) -> None:
        """Diff self.messages against a pre-compression snapshot and append one
        record to compression_events.jsonl. New messages (summaries) are tagged
        with a 'c' uid and stored with full content in the record."""
        import memory as _mem_evt
        for m in self.messages:
            self._evt_tag(m, prefix="c")
        after = self._evt_snapshot(self.messages)
        before_map = dict(before)
        after_map  = dict(after)
        before_uids = [u for u, _ in before]
        after_uids  = [u for u, _ in after]
        dropped  = [u for u in before_uids if u not in after_map]
        added    = [{"uid": u, "message": m} for u, m in zip(after_uids, self.messages) if u not in before_map]
        replaced = []
        for u in after_uids:
            if u in before_map and before_map[u] != after_map[u]:
                replaced.append({
                    "uid": u,
                    "tokens_before": _mem_evt.count_tokens([{"content": before_map[u]}]),
                    "tokens_after":  _mem_evt.count_tokens([{"content": after_map[u]}]),
                    "new_content":   after_map[u],
                })
        self._evt_n_compression += 1
        self._evt_seq += 1
        self._evt_append("compression_events.jsonl", {
            "seq":          self._evt_seq,
            "event_idx":    self._evt_n_compression,
            "step":         self.n_calls,        # calls completed so far; fires before call step+1
            "kind":         kind,
            **fields,
            "tokens_before": _mem_evt.count_tokens([{"content": c} for _, c in before]),
            "tokens_after":  _mem_evt.count_tokens(self.messages),
            "dropped":      dropped,
            "replaced":     replaced,
            "added":        added,
            "before_uids":  before_uids,
            "after_uids":   after_uids,
        })

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                try:
                    self.save(self.config.output_path)
                finally:
                    self._write_token_log()
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def _write_token_log(self) -> None:
        """Flush stats even when a model/tool error interrupts the current step."""
        if os.environ.get("MSWEA_TOKEN_LOG_PATH"):
            import memory
            memory.write_token_log(self)

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages.

        Memory primitive hook
        ---------------------
        Reads two environment variables before every LLM call:
          MSWEA_PRIMITIVE    : "truncation" | "summarization"
                               | "summarization_free" | "structured_summarize_free" (no length target)
          MSWEA_TOKEN_BUDGET : int  — fires when estimated prompt tokens exceed this

        Most primitives target current_tokens * COMPRESSION_RATIO. TRC instead
        clears all but the latest three tool results, then drops oldest complete
        turns only if needed to meet the budget. System and task are protected.

        Token usage and compression stats are accumulated on self._mem_* and written
        to MSWEA_TOKEN_LOG_PATH (if set) after every call.
        """
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

        # ── Memory primitive hook ────────────────────────────────────────────
        #
        # WHAT IS self.messages?
        #   The full conversation history accumulated so far:
        #     messages[0]  — system prompt (never changes)
        #     messages[1]  — user message containing the task (never changes)
        #     messages[2+] — alternating assistant/user messages from each step
        #   On every LLM call (model.query below), the ENTIRE history is sent.
        #   So the history IS the context window.
        #
        # WHAT IS THE BUDGET?
        #   MSWEA_TOKEN_BUDGET is a token count threshold for the context window.
        #   It is set by run_experiment.py as:
        #     budget = max(step_prompt_tokens from baseline run) * budget_pct
        #   e.g. if the baseline peak context was 40K and budget_pct=0.60,
        #   budget = 24K.  Compression fires when the history exceeds 24K tokens.
        #   At p100 (baseline) the budget is set to 999999999 so it never fires.
        #
        # TRIGGER: context window size > budget
        #   We measure the current history size with count_tokens(self.messages)
        #   using tiktoken (cl100k_base, accurate to ~5% across models).
        #   This is checked BEFORE each LLM call. B is a compression trigger,
        #   not a hard limit: protected content or a summary may still exceed it.
        #   No reset is needed; the next call checks the current history again.
        #
        # WHAT COMPRESSION DOES:
        #   Both primitives protect messages[0:2] (system + task) — never touched.
        #   They operate only on messages[2:] (the agent's working history).
        #   Standalone TR targets budget * COMPRESSION_RATIO; TRC targets budget.
        #   Other policies retain their current_size * COMPRESSION_RATIO target.
        #
        #   truncation  — drops oldest complete assistant/result turns until size
        #                 <= target, preserving the latest turn. No extra LLM call.
        #   summarization — one extra LLM call produces a structured summary of
        #                 messages[2:], which replaces the entire compressible
        #                 window with a single summary message.
        #
        _adaptive_config = self._memory_config
        _adaptive_events = []
        if _adaptive_config is not None:
            _primitive = _adaptive_config.primitive
            _budget = _adaptive_config.budget
        else:
            _primitive = os.environ.get("MSWEA_PRIMITIVE", "")
            _budget = int(os.environ.get("MSWEA_TOKEN_BUDGET", "0") or "0")

        # ── ProbeCtrl full per-step context logging (additive; inert unless the
        #    MSWEA_FULL_CONTEXT_LOG_DIR env var is set). Captures the exact context
        #    sent to the model each step, plus the pre-compression context whenever
        #    a compression event fires, so every event's full-vs-compressed pair is
        #    reconstructable with a Δ=0 checksum against the recorded prompt_tokens. ──
        _pc_dir   = os.environ.get("MSWEA_FULL_CONTEXT_LOG_DIR", "")
        _pc_pre   = None     # pre-compression context snapshot (set iff compression fires this step)
        _pc_fired = False

        # ── Online TRC hook ──────────────────────────────────────────────────
        # Online TRC: freeze-window clearing.
        # Protect the newest FREEZE_K tool results verbatim; every step, clear
        # the newest tool result *outside* that window (≈ the result from step
        # n-FREEZE_K-1 when the history is regular).
        #
        # The target is selected by ROLE, not by position.  An earlier version
        # rewrote messages[-9] unconditionally; a FormatError (user message with
        # no assistant reply), a truncation that drops an odd number of messages,
        # or a summary insertion all shift message parity, and the fixed offset
        # then landed on assistant messages or — when len(messages)==10 — on
        # messages[1], the task statement itself.  Candidates here are user-role
        # messages in the compressible window (index ≥ N_PROTECTED) that are not
        # summaries and not already-cleared stubs, so the task statement and
        # assistant turns can never be selected.
        _FREEZE_K = _adaptive_config.freeze_k if _adaptive_config is not None else 4
        _OTRC_FAMILY = (
            "online_trc",
            "online_trc_summarize_partial",
            "online_trc_structured_summarize_partial",
        )
        _step_interval = _adaptive_config.step_interval if _adaptive_config is not None else None
        _online_due = (_step_interval is None
                       or self.n_calls - self._memory_online_step >= _step_interval)
        if _primitive in _OTRC_FAMILY and _online_due:
            import memory as _mem_otrc

            _OTRC_STUB       = "[tool-result cleared"
            _OTRC_SKIP_PREFIX = (_OTRC_STUB, "[TOOL OUTPUT CLEARED", "[CONTEXT SUMMARY", "[COMPRESSED HISTORY")

            def _otrc_clearable(msg: dict) -> bool:
                # Summaries are identified the same way as in the offline TRC
                # primitives (extra["kind"] tag first, marker prefix for old runs).
                if msg.get("role") != "user" or _mem_otrc.is_summary_message(msg):
                    return False
                content = msg.get("content")
                if not isinstance(content, str):
                    return False
                return not content.startswith(_OTRC_SKIP_PREFIX)

            _otrc_candidates = [
                i for i in range(_mem_otrc.N_PROTECTED, len(self.messages))
                if _otrc_clearable(self.messages[i])
            ]
            if len(_otrc_candidates) > _FREEZE_K:
                if _adaptive_config is not None:
                    _online_before = _mem_otrc.count_tokens(self.messages)
                _result_idx   = _otrc_candidates[-(_FREEZE_K + 1)]
                _result_msg   = self.messages[_result_idx]
                _orig_tokens  = _mem_otrc.count_tokens([_result_msg])
                _step_cleared = self.n_calls - (_FREEZE_K + 1)   # approximate label (regular history)
                _new_content  = f"[tool-result cleared — online-trc — {_orig_tokens} tok — step {_step_cleared}]"
                _evt_before_otrc = self._evt_snapshot(self.messages) if self._evt_dir else None
                self.messages[_result_idx] = {**_result_msg, "content": _new_content}
                if _evt_before_otrc is not None:
                    self._evt_record_compression(
                        "online_trc", _evt_before_otrc, primitive=_primitive,
                        flag_from_step=_step_cleared, target_tokens=None, budget=_budget,
                        cleared_index=_result_idx,
                        **({"adaptive_config": _adaptive_config.to_dict()}
                           if _adaptive_config is not None else {}),
                    )

                _tokens_saved_otrc = max(0, _orig_tokens - _mem_otrc.count_tokens([self.messages[_result_idx]]))
                self._mem_online_trc_flags.append({
                    "step":           self.n_calls,
                    "flag_from_step": _step_cleared,
                    "tokens_cleared": _tokens_saved_otrc,
                })
                self._mem_online_trc_tokens_saved += _tokens_saved_otrc
                _mem_otrc.write_token_log(self)
                if _adaptive_config is not None:
                    _adaptive_events.append({
                        "kind": "online_trc", "tokens_before": _online_before,
                        "tokens_after": _mem_otrc.count_tokens(self.messages),
                    })
            elif _step_interval is not None:
                # The scheduled trigger still fires with no eligible result.
                # This lets the policy select its next action without waiting
                # for freeze-window eligibility (or losing the scheduled event).
                _online_tokens = _mem_otrc.count_tokens(self.messages)
                # No history changed: keep this trigger only in adaptive logs,
                # so online_trc compression records continue to mean actual clears.
                _adaptive_events.append({
                    "kind": "online_trc", "tokens_before": _online_tokens,
                    "tokens_after": _online_tokens, "skipped_reason": "no_eligible_result",
                })
        # ── End online TRC hook ──────────────────────────────────────────────

        if _primitive and _budget is not None and _budget > 0:
            import memory as _mem   # agentCtx root must be on PYTHONPATH
            _depth = _adaptive_config.depth if _adaptive_config is not None else _mem.COMPRESSION_RATIO
            # Measure the current context window size (= full history size).
            # This is what the model would receive on the next call.
            _current = _mem.count_tokens(self.messages)
            if _current > _budget:
                # History has grown past the budget — compress it now.
                # Standalone TR and length-free summary fallbacks share B*r.
                # Other proportional policies retain their current-size targets.
                _free_summary = _primitive in ("summarization_free", "structured_summarize_free")
                if _primitive == "tool_result_clear":
                    _target = _budget
                elif _primitive == "truncation" or _free_summary:
                    _target = max(1, int(_budget * _depth))
                else:
                    _target = max(1, int(_current * _depth))
                _evt_before = self._evt_snapshot(self.messages) if self._evt_dir else None
                _evt_sum_pt0 = self._mem_summarization_prompt_tokens
                _evt_sum_lat0 = self._mem_summarization_latency_s
                _trc_fallback = False
                _trc_stats = {}
                _evt_picked   = None
                if hasattr(_mem, "pop_summary_outcome"):
                    _mem.pop_summary_outcome()   # discard anything stale before this event
                if _pc_dir:
                    import copy as _pc_copy
                    _pc_pre   = _pc_copy.deepcopy(self.messages)  # full context entering compression
                    _pc_fired = True
                if _primitive == "summarization":
                    # LLM call produces a structured summary replacing messages[2:].
                    # Tokens used by that summary call are tracked separately so we
                    # can distinguish them from the main agent's token usage.
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.summarize(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens              += _pt
                    self._mem_completion_tokens          += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s    += _sum_lat
                elif _primitive == "structured_summarize":
                    # LLM call produces a schema-guided summary (Task / Files Modified /
                    # Files Examined / Execution Anchors / Current State).
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.structured_summarize(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "summarization_free":
                    # SU-free: same as "summarization" but the summarizer gets no word
                    # target (depth-invariant); _target only sizes the complete-turn TR fallback.
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.summarize_free(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "structured_summarize_free":
                    # SS-free: schema-guided summary with no word target (depth-invariant).
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.structured_summarize_free(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "summarization_partial":
                    # SU-partial: summarize the head, keep the budget-fitting tail verbatim.
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.summarize_partial(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "structured_summarize_partial":
                    # SS-partial: structured-summarize the head, keep budget-fitting tail verbatim.
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.structured_summarize_partial(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "tool_result_clear":
                    # Clear ALL results older than the latest three, then drop
                    # complete oldest turns only until the budget is met.
                    self.messages, _saved, _trc_fallback = _mem.tool_result_clear(
                        self.messages, _budget, stats=_trc_stats
                    )
                    if _trc_fallback:
                        self._mem_trc_fallback_events += 1
                elif _primitive == "scored_tool_result_clear":
                    # Ranked clearing: stubs out bash output bodies lowest-score first.
                    # Score = type_weight × size + citation_boost (ACT-R inspired).
                    # Falls back to truncate() if scored clearing is insufficient.
                    self.messages, _saved, _trc_fallback = _mem.scored_tool_result_clear(self.messages, _target)
                    if _trc_fallback:
                        self._mem_trc_fallback_events += 1
                elif _primitive == "trc_summarize":
                    # Stage 1: clear every result older than the newest three;
                    # the summary stage handles any remaining budget overflow.
                    self.messages, _saved, _ = _mem.tool_result_clear(
                        self.messages, _budget, fallback_truncate=False, stats=_trc_stats
                    )
                    # Stage 2: if still over budget, summarize the remaining history.
                    _after_trc = _mem.count_tokens(self.messages)
                    if _after_trc > _budget:
                        _target2 = max(1, int(_after_trc * _depth))
                        self.messages, _saved2, _pt, _ct, _sum_lat = _mem.summarize(
                            self.messages, self.model, _target2
                        )
                        _saved += _saved2
                        self._mem_prompt_tokens               += _pt
                        self._mem_completion_tokens           += _ct
                        self._mem_summarization_prompt_tokens += _pt
                        self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "trc_structured_summarize":
                    # Stage 1: clear every result older than the newest three;
                    # the summary stage handles any remaining budget overflow.
                    self.messages, _saved, _ = _mem.tool_result_clear(
                        self.messages, _budget, fallback_truncate=False, stats=_trc_stats
                    )
                    # Stage 2: if still over budget, structured-summarize the remaining history.
                    _after_trc = _mem.count_tokens(self.messages)
                    if _after_trc > _budget:
                        _target2 = max(1, int(_after_trc * _depth))
                        self.messages, _saved2, _pt, _ct, _sum_lat = _mem.structured_summarize(
                            self.messages, self.model, _target2
                        )
                        _saved += _saved2
                        self._mem_prompt_tokens               += _pt
                        self._mem_completion_tokens           += _ct
                        self._mem_summarization_prompt_tokens += _pt
                        self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "online_trc":
                    # OTRC+TR: freeze window already cleared the oldest-outside-window result above;
                    # truncation fires here as the budget-time fallback.
                    self.messages, _saved = _mem.truncate(self.messages, _target)
                elif _primitive == "online_trc_summarize_partial":
                    # OTRC+SU-partial: freeze window already cleared a result above;
                    # SU-partial fires as the budget-time fallback (head summarized,
                    # budget-fitting tail kept verbatim — preserves OTRC's freeze window).
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.summarize_partial(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "online_trc_structured_summarize_partial":
                    # OTRC+SS-partial: same as OTRC+SU-partial but with structured summary.
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.structured_summarize_partial(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive in ("staggered_alternate", "staggered_random"):
                    # Staggered: at each compression event, pick one of the
                    # oracle-optimal pair (TR + budget-best). Pair is fixed by budget.
                    _STAGGERED_PAIRS = {
                        10000: ("truncation", "trc_structured_summarize"),
                        15000: ("truncation", "summarization_partial"),
                        20000: ("truncation", "trc_structured_summarize"),
                    }
                    if _budget not in _STAGGERED_PAIRS:
                        raise RuntimeError(
                            f"staggered: no pair defined for budget {_budget}; "
                            f"add an entry in _STAGGERED_PAIRS"
                        )
                    _p1, _p2 = _STAGGERED_PAIRS[_budget]

                    _idx = getattr(self, "_staggered_event_idx", 0)
                    if _primitive == "staggered_alternate":
                        _picked = _p1 if _idx % 2 == 0 else _p2
                    else:  # staggered_random
                        if not hasattr(self, "_staggered_rng"):
                            import random as _stg_random
                            self._staggered_rng = _stg_random.Random(
                                hash(os.environ.get("MSWEA_RUN_KEY", ""))
                            )
                        _picked = _p1 if self._staggered_rng.random() < 0.5 else _p2

                    if not hasattr(self, "_staggered_log"):
                        self._staggered_log = []
                    self._staggered_log.append(_picked)
                    _evt_picked = _picked

                    # Dispatch to the picked underlying primitive.
                    if _picked == "truncation":
                        self.messages, _saved = _mem.truncate(self.messages, _target)
                    elif _picked == "summarization_partial":
                        self.messages, _saved, _pt, _ct, _sum_lat = _mem.summarize_partial(
                            self.messages, self.model, _target
                        )
                        self._mem_prompt_tokens               += _pt
                        self._mem_completion_tokens           += _ct
                        self._mem_summarization_prompt_tokens += _pt
                        self._mem_summarization_latency_s     += _sum_lat
                    elif _picked == "trc_structured_summarize":
                        # TRC stage (no truncate fallback).
                        self.messages, _saved, _ = _mem.tool_result_clear(
                            self.messages, _budget, fallback_truncate=False, stats=_trc_stats
                        )
                        # SS fallback if still over budget.
                        _after_trc = _mem.count_tokens(self.messages)
                        if _after_trc > _budget:
                            _target2 = max(1, int(_after_trc * _depth))
                            self.messages, _saved2, _pt, _ct, _sum_lat = _mem.structured_summarize(
                                self.messages, self.model, _target2
                            )
                            _saved += _saved2
                            self._mem_prompt_tokens               += _pt
                            self._mem_completion_tokens           += _ct
                            self._mem_summarization_prompt_tokens += _pt
                            self._mem_summarization_latency_s     += _sum_lat
                    else:
                        raise RuntimeError(f"staggered: unknown picked primitive {_picked}")

                    self._staggered_event_idx = _idx + 1
                elif _primitive == "truncation":
                    self.messages, _saved = _mem.truncate_oldest_turns(self.messages, _target)
                else:  # legacy default for unrecognized primitive names
                    # Drop oldest messages from messages[2:] until size <= target.
                    self.messages, _saved = _mem.truncate(self.messages, _target)

                # Outcome of the summary request made by this event (None when the
                # primitive did not request one, e.g. TRC stage 1 was sufficient).
                _sum_outcome = _mem.pop_summary_outcome() if hasattr(_mem, "pop_summary_outcome") else None
                if _sum_outcome is not None:
                    self._mem_summary_outcomes.append({
                        "step":       self.n_calls,
                        "primitive":  _primitive,
                        "picked":     _evt_picked,
                        "attempts":   _sum_outcome.get("attempts"),
                        "accepted":   _sum_outcome.get("accepted"),
                        "rejections": _sum_outcome.get("rejections", []),
                        "fallback":   _sum_outcome.get("fallback"),
                        # flags of the accepted (or last rejected) reply:
                        # markers, finish_reason, raw_chars (audit trail)
                        "flags":      _sum_outcome.get("flags"),
                    })

                if _trc_stats:
                    self._mem_trc_events.append({
                        "step": self.n_calls, "primitive": _primitive,
                        "picked": _evt_picked, **_trc_stats,
                    })

                _after = _mem.count_tokens(self.messages)
                _tr_stats = None
                if _primitive == "truncation" or (
                    _free_summary and _sum_outcome is not None and _sum_outcome.get("fallback") == "truncate"
                ):
                    _tr_stats = {
                        "policy": "budget_ratio_complete_turns_v1",
                        "step": self.n_calls,
                        "primitive": _primitive,
                        "budget_tokens": _budget,
                        "target_tokens": _target,
                        "tokens_before": _current,
                        "tokens_after": _after,
                        "tokens_saved": _current - _after,
                        # Independent flags: an event can satisfy more than one.
                        "target_not_met": _after > _target,
                        "budget_exceeded": _after > _budget,
                        "zero_reduction": _after == _current,
                    }
                    self._mem_tr_events.append(_tr_stats)

                if _evt_before is not None:
                    self._evt_record_compression(
                        "budget", _evt_before, primitive=_primitive, picked=_evt_picked,
                        budget=_budget, target_tokens=_target, trc_fallback=bool(_trc_fallback),
                        tokens_saved_reported=_saved,
                        summary_prompt_tokens=self._mem_summarization_prompt_tokens - _evt_sum_pt0,
                        summary_latency_s=self._mem_summarization_latency_s - _evt_sum_lat0,
                        summary_outcome=_sum_outcome,
                        trc_stats=_trc_stats or None,
                        **({"adaptive_config": _adaptive_config.to_dict()}
                           if _adaptive_config is not None else {}),
                        **({"tr_stats": _tr_stats} if _tr_stats is not None else {}),
                    )

                # Record event metadata for the token log.
                if _current > 0:
                    self._mem_compression_ratios.append(_after / _current)
                self._mem_compression_events += 1
                self._mem_tokens_saved       += _saved
                self._mem_compression_event_steps.append(self.n_calls)
                self._mem_context_tokens_at_compression.append(_current)
                self._mem_context_tokens_after_compression.append(_after)
                # Persist EVERY compression before the next model call. The
                # caller may fail, hang, or be killed before a response arrives.
                # The trajectory is flushed too: run() only saves it in the
                # step's finally, which never runs when the harness SIGKILLs
                # the process on timeout, so trajectory.json used to lag the
                # event log and token log by one compression in such runs.
                self._write_token_log()
                if self.config.output_path:
                    self.save(self.config.output_path)
                if _adaptive_config is not None:
                    _adaptive_events.append({
                        "kind": "budget", "tokens_before": _current, "tokens_after": _after,
                    })
                # No reset of _mem_prompt_tokens needed: the trigger checks
                # current context size directly. If it still exceeds B, another
                # compression event can fire before the next model call.
        # ────────────────────────────────────────────────────────────────────

        self._advance_memory_policy(_adaptive_config, _adaptive_events)
        self.n_calls += 1
        if _pc_dir:
            import copy as _pc_copy2
            _pc_sent = _pc_copy2.deepcopy(self.messages)  # exact context sent to the model this step
        _t0 = time.time()
        _query_error = None
        try:
            message = self.model.query(self.messages)
        except (Exception, KeyboardInterrupt) as exc:
            # FormatError carries the provider response even though parsing
            # failed. Transport errors may have no response/usage at all.
            _query_error = exc
            message = getattr(exc, "model_response", None) or {}
        _latency = time.time() - _t0

        self.cost += message.get("extra", {}).get("cost", 0.0)
        if _query_error is None:
            self.add_messages(message)
        elif isinstance(_query_error, FormatError) and _query_error.messages:
            # Keep the rejected response (including usage, reasoning and finish
            # reason) in feedback metadata. API preparation strips extra, so it
            # does not turn the rejected response into a conversation action.
            feedback, *rest = _query_error.messages
            _query_error.messages = ({
                **feedback,
                "extra": {**feedback.get("extra", {}), **message.get("extra", {}),
                          "model_call_step": self.n_calls},
            }, *rest)

        # ── Accumulate token usage and write log ─────────────────────────────
        _extra = message.get("extra", {})
        _resp  = _extra.get("response", {})
        _usage = (_resp.get("usage") or {}) if isinstance(_resp, dict) else {}
        _step_pt = _usage.get("prompt_tokens", 0) or 0
        _step_ct = _usage.get("completion_tokens", 0) or 0
        self._mem_prompt_tokens     += _step_pt
        self._mem_completion_tokens += _step_ct
        self._mem_total_latency     += _latency
        self._mem_call_latencies.append(_latency)
        self._mem_step_prompt_tokens.append(_step_pt)
        self._mem_step_completion_tokens.append(_step_ct)
        _call_status = (
            "ok" if _query_error is None
            else "format_error" if isinstance(_query_error, FormatError) else "error"
        )
        # Legacy arrays use zero when no usage is returned. These records make
        # that absence explicit rather than claiming the failed call was free.
        try:
            from agentctx.compression.cache_metrics import cache_usage
        except ImportError:   # agentCtx not on PYTHONPATH: keep the record, skip cache fields
            _cache_fields = {"cache_usage_status": "collector_unavailable"}
        else:
            _cache_fields = cache_usage(_usage)
        self._mem_model_call_records.append({
            "step": self.n_calls, "status": _call_status,
            "error_type": type(_query_error).__name__ if _query_error is not None else None,
            "prompt_tokens": _usage.get("prompt_tokens"),
            "completion_tokens": _usage.get("completion_tokens"),
            "latency_s": round(_latency, 3),
            **_cache_fields,
        })
        if _pc_dir:
            import json as _pc_json
            from pathlib import Path as _PcPath
            _pc_rec = {
                "step": self.n_calls,                      # 1-based call index (matches step_prompt_tokens order)
                "recorded_prompt_tokens": _step_pt,        # for Δ=0 checksum of sent_context
                "compressed_this_step": _pc_fired,
                "sent_context": _pc_sent,                  # exact context the model saw this step (post-compression)
                "pre_compression_context": _pc_pre,        # full context before compression (None unless fired)
                "response_text": message.get("content", ""),  # empty for failures without a response
                "response_status": _call_status,
            }
            _pcd = _PcPath(_pc_dir)
            _pcd.mkdir(parents=True, exist_ok=True)
            with open(_pcd / "full_context_log.jsonl", "a") as _pcf:
                _pcf.write(_pc_json.dumps(_pc_rec) + "\n")
        self._write_token_log()
        # ────────────────────────────────────────────────────────────────────

        if _query_error is not None:
            raise _query_error
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
                "staggered_log": getattr(self, "_staggered_log", []),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
