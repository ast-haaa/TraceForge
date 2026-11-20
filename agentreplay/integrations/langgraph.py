# agentreplay/integrations/langgraph.py
"""LangGraph integration for AgentReplay.

Provides AgentReplayHandler, a LangChain/LangGraph BaseCallbackHandler
that records steps, tool calls, and LLM calls incrementally into AgentReplay
without manual rec.start_step or record_event calls in the agent code.
"""
from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from agentreplay.models import EventKind, EventSource, Step, TraceRunType, TraceStatus
from agentreplay.sdk import Recorder, _active_replay_session


class AgentReplayHandler(BaseCallbackHandler):
    """LangGraph / LangChain callback handler for AgentReplay.

    Usage::

        with record_trace(...) as rec:
            graph.invoke(inputs, config={"callbacks": [AgentReplayHandler(rec)]})
    """

    def __init__(self, recorder: Recorder) -> None:
        super().__init__()
        self.rec = recorder
        # Maps run_id -> open step data dict
        self._open_steps: dict[Any, dict[str, Any]] = {}
        # Maps run_id -> Step object for fast lookup
        self._step_by_run_id: dict[Any, Step] = {}
        # Maps child_run_id -> parent_run_id to traverse execution hierarchy
        self._parent_map: dict[Any, Any] = {}
        # Maps run_id -> open tool call data dict
        self._open_tools: dict[Any, dict[str, Any]] = {}
        # Maps run_id -> open LLM call data dict
        self._open_llms: dict[Any, dict[str, Any]] = {}

    def _determine_source(self, step: Step | None = None) -> EventSource:
        """Determine event source (recorded, replayed, mocked, or live)."""
        if _active_replay_session.get() is not None:
            return EventSource.REPLAYED
        if self.rec.trace.run_type == TraceRunType.FORK:
            fork_step = self.rec.trace.fork_step
            if fork_step is not None and step is not None and step.index == fork_step:
                return EventSource.MOCKED
            return EventSource.LIVE
        return EventSource.RECORDED

    def _find_step_for_run(self, run_id: Any) -> Step | None:
        """Climb parent hierarchy to find the enclosing node Step."""
        curr = run_id
        while curr is not None:
            if curr in self._step_by_run_id:
                return self._step_by_run_id[curr]
            curr = self._parent_map.get(curr)
        return self.rec._current_step

    def _find_step_info_for_run(self, run_id: Any) -> dict[str, Any] | None:
        """Climb parent hierarchy to find the enclosing node step_info dict."""
        curr = run_id
        while curr is not None:
            if curr in self._open_steps:
                return self._open_steps[curr]
            curr = self._parent_map.get(curr)
        return None

    # ------------------------------------------------------------------
    # Graph Node (Chain) Callbacks -> Step
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is not None:
            self._parent_map[run_id] = parent_run_id

        meta = metadata or {}
        node_name = meta.get("langgraph_node")

        # Skip LangGraph root graph and internal control nodes
        if not node_name or node_name in ("__start__", "__end__", "LangGraph"):
            return None

        # Real graph node executions in LangGraph are tagged with 'graph:step:N'.
        # Internal conditional branch evaluations are tagged with 'seq:step:N' and should not form Steps.
        tag_list = tags or []
        if tag_list and not any(t.startswith("graph:step:") for t in tag_list):
            return None

        # Clean state_before: drop internal LangGraph keys starting with __
        raw_state = inputs if isinstance(inputs, dict) else {"input": inputs}
        safe_state_before = {
            k: v for k, v in raw_state.items() if not str(k).startswith("__")
        }

        step = self.rec.start_step(node_name=node_name, state_before=safe_state_before)
        self._open_steps[run_id] = {
            "step": step,
            "node_name": node_name,
            "start_time": time.perf_counter(),
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }
        self._step_by_run_id[run_id] = step
        return None

    def on_chain_end(
        self,
        outputs: dict[str, Any] | Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        step_info = self._open_steps.pop(run_id, None)
        if step_info is not None:
            step = step_info["step"]
            latency = (time.perf_counter() - step_info["start_time"]) * 1000

            raw_out = outputs if isinstance(outputs, dict) else {"output": outputs}
            safe_state_after = {
                k: v for k, v in raw_out.items() if not str(k).startswith("__")
            }

            self.rec.end_step(
                state_after=safe_state_after,
                model=step_info.get("model"),
                input_tokens=step_info.get("input_tokens", 0),
                output_tokens=step_info.get("output_tokens", 0),
                cost=step_info.get("cost", 0.0),
                latency_ms=latency,
                step=step,
            )
            return None

        # If root graph finishes (parent_run_id is None) and trace is still RECORDING:
        if parent_run_id is None and self.rec.trace.status == TraceStatus.RECORDING:
            final_out = outputs if isinstance(outputs, dict) else {"result": outputs}
            safe_final = {
                k: v for k, v in final_out.items() if not str(k).startswith("__")
            }
            self.rec.finalize(final_output=safe_final)
        return None

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        step_info = self._open_steps.pop(run_id, None)
        if step_info is not None:
            step = step_info["step"]
            latency = (time.perf_counter() - step_info["start_time"]) * 1000
            self.rec.end_step(
                state_after={"error": str(error)},
                latency_ms=latency,
                step=step,
            )

        self.rec.trace.status = TraceStatus.FAILED
        self.rec.db.flush()
        self.rec.db.commit()
        return None

    # ------------------------------------------------------------------
    # Tool Call Callbacks -> TOOL_CALL Event
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str | Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is not None:
            self._parent_map[run_id] = parent_run_id

        tool_name = (serialized or {}).get("name", "unknown_tool")
        if inputs is not None and isinstance(inputs, dict):
            args = inputs
        elif isinstance(input_str, dict):
            args = input_str
        elif isinstance(input_str, str):
            try:
                args = json.loads(input_str)
            except Exception:
                args = input_str
        else:
            args = input_str

        tool_args = args if args else None
        raw_input = {"name": tool_name, "args": tool_args}

        self._open_tools[run_id] = {
            "parent_run_id": parent_run_id,
            "tool_name": tool_name,
            "raw_input": raw_input,
            "start_time": time.perf_counter(),
        }
        return None

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_info = self._open_tools.pop(run_id, None)
        if tool_info is None:
            return None

        p_id = parent_run_id or tool_info.get("parent_run_id")
        step = self._find_step_for_run(p_id)
        if step is None:
            return None

        ev_source = self._determine_source(step)
        self.rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_info["raw_input"],
            output=output,
            source=ev_source,
            step=step,
        )
        return None

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_info = self._open_tools.pop(run_id, None)
        raw_input = (
            tool_info["raw_input"]
            if tool_info
            else {"name": "unknown_tool", "args": None}
        )
        p_id = parent_run_id or (
            tool_info.get("parent_run_id") if tool_info else None
        )
        step = self._find_step_for_run(p_id)

        if step is not None:
            ev_source = self._determine_source(step)
            self.rec.record_event(
                kind=EventKind.TOOL_CALL,
                raw_input=raw_input,
                output=None,
                error=str(error),
                source=ev_source,
                step=step,
            )

        self.rec.trace.status = TraceStatus.FAILED
        self.rec.db.flush()
        self.rec.db.commit()
        return None

    # ------------------------------------------------------------------
    # LLM / ChatModel Callbacks -> LLM_CALL Event
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is not None:
            self._parent_map[run_id] = parent_run_id
        prompt_text = prompts[0] if prompts else ""
        self._open_llms[run_id] = {
            "parent_run_id": parent_run_id,
            "prompt": prompt_text,
            "serialized": serialized,
            "start_time": time.perf_counter(),
        }
        return None

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is not None:
            self._parent_map[run_id] = parent_run_id
        prompt_text = ""
        if messages and messages[0]:
            prompt_text = "\n".join(
                getattr(m, "content", str(m)) for m in messages[0]
            )
        self._open_llms[run_id] = {
            "parent_run_id": parent_run_id,
            "prompt": prompt_text,
            "serialized": serialized,
            "start_time": time.perf_counter(),
        }
        return None

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        llm_info = self._open_llms.pop(run_id, None)
        if llm_info is None:
            return None

        p_id = parent_run_id or llm_info.get("parent_run_id")
        step = self._find_step_for_run(p_id)
        if step is None:
            return None

        llm_out = getattr(response, "llm_output", {}) or {}

        # Under a replay session, preserve recorded latency so event outputs match 100%
        if _active_replay_session.get() is not None and llm_out.get("latency_ms") is not None:
            latency = float(llm_out["latency_ms"])
        else:
            latency = (time.perf_counter() - llm_info["start_time"]) * 1000

        text = ""
        if hasattr(response, "generations") and response.generations:
            first_gen = response.generations[0][0]
            text = getattr(first_gen, "text", "") or getattr(
                getattr(first_gen, "message", None), "content", ""
            )
        model_name = llm_out.get("model") or llm_out.get("model_name") or "stub-llm"
        token_usage = llm_out.get("token_usage") or {}
        in_tokens = llm_out.get("input_tokens") or token_usage.get("prompt_tokens") or 0
        out_tokens = (
            llm_out.get("output_tokens") or token_usage.get("completion_tokens") or 0
        )
        tot_tokens = in_tokens + out_tokens
        cost = llm_out.get("cost", 0.0)

        # Parse decision and reason if present in text
        decision = llm_out.get("decision")
        reason = llm_out.get("reason", "")
        if not decision:
            if "DECISION: approve" in text:
                decision = "approve"
            elif "DECISION: deny" in text:
                decision = "deny"
            else:
                decision = "approve"
        if not reason:
            if "REASON:" in text:
                reason = text.split("REASON:", 1)[1].strip()
            else:
                reason = text

        out_dict = {
            "response": text,
            "decision": decision,
            "reason": reason,
            "model": model_name,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": tot_tokens,
            "latency_ms": latency,
            "cost": cost,
        }

        ev_source = self._determine_source(step)
        self.rec.record_event(
            kind=EventKind.LLM_CALL,
            raw_input={"prompt": llm_info["prompt"]},
            output=out_dict,
            source=ev_source,
            step=step,
        )

        step_info = self._find_step_info_for_run(p_id)
        if step_info is not None:
            step_info["model"] = model_name
            step_info["input_tokens"] += in_tokens
            step_info["output_tokens"] += out_tokens
            step_info["cost"] += cost
        return None

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        llm_info = self._open_llms.pop(run_id, None)
        p_id = parent_run_id or (llm_info.get("parent_run_id") if llm_info else None)
        step = self._find_step_for_run(p_id)
        prompt = llm_info["prompt"] if llm_info else ""

        if step is not None:
            ev_source = self._determine_source(step)
            self.rec.record_event(
                kind=EventKind.LLM_CALL,
                raw_input={"prompt": prompt},
                output=None,
                error=str(error),
                source=ev_source,
                step=step,
            )

        self.rec.trace.status = TraceStatus.FAILED
        self.rec.db.flush()
        self.rec.db.commit()
        return None
