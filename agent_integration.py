"""Plugging focus_beam into the Homework 1, Task 6 graph.

Only the executor changes: it gets focus_beam next to its coding tool, and a
system prompt that lets it pick the settings itself. Replace the two imports
marked HW1 with the names from your own Task 6 code.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from focus_tool import focus_beam_tool

# HW1: your existing objects
from hw1_task6 import llm, coding_tool, build_graph  # noqa: E402

EXECUTOR_TOOLS = [coding_tool, focus_beam_tool]
TOOLS_BY_NAME = {t.name: t for t in EXECUTOR_TOOLS}

EXECUTOR_PROMPT = """You are the executor. You run tools; you never ask the user for settings.
For a request to focus the beam, use focus_beam on the simulated DMD bench.
DMD is 32x32 mirrors, camera 64x64 px, speckle grain about 2 px.
Plan a short campaign on your own:
  1. Start with a coarse, fast run (large macropixels) to check the bench works.
  2. Decrease macropixel_size (more degrees of freedom -> higher enhancement,
     theory for binary amplitude ~ N/(2*pi)), and give more iterations when
     there are more macropixels (at least ~5x the number of macropixels).
  3. Try at least one change to the region sizes (focus_size, background ring)
     and judge whether it helped.
Pass a one-sentence `reason` with every call. After each result decide the next
settings from the numbers returned. Stop after 4-6 runs and report a table of
all runs and the best enhancement, which is the one to keep."""

MAX_TOOL_STEPS = 8


def executor_node(state):
    """Executor: model call with tools, loops until it stops calling tools."""
    model = llm.bind_tools(EXECUTOR_TOOLS)
    msgs = [SystemMessage(EXECUTOR_PROMPT)] + list(state["messages"])
    new = []
    best = state.get("best_enhancement")
    for _ in range(MAX_TOOL_STEPS):
        ai: AIMessage = model.invoke(msgs + new)
        ai.name = "executor"
        new.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            out = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            new.append(ToolMessage(content=str(out), tool_call_id=call["id"], name=call["name"]))
            if call["name"] == "focus_beam":
                eta = json.loads(out)["enhancement"]
                if best is None or eta > best["enhancement"]:
                    best = json.loads(out)          # the agent keeps the best result
    return {"messages": new, "best_enhancement": best}


if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph(executor_node=executor_node, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "focus-1"}}
    out = graph.invoke({"messages": [("user", "Focus the beam on the bench.")]}, cfg)
    for m in out["messages"]:
        print(f"[{getattr(m, 'name', None) or m.type}] {m.content}")
    print("kept:", out.get("best_enhancement"))
