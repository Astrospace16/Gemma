"""
agent_adk.py: Runs a coding agent on a SciSilentBench task using Google's ADK.

This module defines the async function `run_agent_on_task`, which:
  1. Sets up an in-memory session and a LiteLLM-backed Gemma 4 agent.
  2. Gives the agent three tools: read_file, run_pytest, run_python.
  3. Runs the agent under either Condition A (baseline) or Condition B (invariant + abstention).
  4. Returns a structured result dict with the final patch, decision, confidence, and cost metrics.
"""

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from mutator import run_code_check


# ---------------------------------------------------------------------------
# Tool implementations (called by the agent)
# ---------------------------------------------------------------------------

def read_file(path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read a range of lines from a file in the repo."""
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    lines = p.read_text().splitlines()
    if end_line == -1 or end_line > len(lines):
        end_line = len(lines)
    return "\n".join(lines[start_line - 1 : end_line])


def run_pytest(test_path: str) -> str:
    """Run pytest on a given test file path and return the output."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nRETURNCODE: {result.returncode}"
    except subprocess.TimeoutExpired:
        return "ERROR: pytest timed out after 120 seconds"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def run_python(script_content: str) -> str:
    """Execute a Python script in the repo environment and return its output."""
    tmp_path = Path("_agent_scratch.py")
    tmp_path.write_text(script_content)
    try:
        result = subprocess.run(
            ["python", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nRETURNCODE: {result.returncode}"
    except subprocess.TimeoutExpired:
        return "ERROR: script timed out after 120 seconds"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Prompt builders for Condition A and Condition B
# ---------------------------------------------------------------------------

CONDITION_A_SYSTEM_PROMPT = """You are a coding agent fixing a silent physical bug in a scientific Python file.
Use the complete buggy source, task description, and visible test supplied in the user message. Identify the physical/unit error and produce the complete corrected Python file with no Markdown fences.
You MUST submit your final decision through the submit_patch tool with the complete source and confidence from 0.0 to 1.0. If you cannot verify a correct fix, call abstain with a reason and confidence. Do not finish with a prose-only answer. At most 10 tool calls.
"""

CONDITION_B_SYSTEM_PROMPT = """You are a coding agent fixing a silent physical bug in a scientific Python file.
Use the complete buggy source, task description, and visible test supplied in the user message. Identify the physical/unit error and produce the complete corrected Python file with no Markdown fences.
Before deciding, run ONE short non-degenerate reference-value invariant using run_python. The script should contain the proposed source and one assert against a known physical value. Then run check_visible_test on the complete proposed source.
If both checks pass, submit your final decision through submit_patch with the complete source and confidence 0.0 to 1.0. If either check fails or you cannot verify, call abstain with a reason and confidence. Do not finish with a prose-only answer. At most 10 tool calls.
"""


def _make_litellm_model(model_name: str) -> LiteLlm:
    """Build a LiteLLM model, mapping Ollama IDs to its chat provider."""
    if model_name.startswith("ollama/"):
        model_name = "ollama_chat/" + model_name.removeprefix("ollama/")
    elif model_name.startswith("hf.co/"):
        model_name = "ollama_chat/" + model_name
    return LiteLlm(model=model_name)

def _build_user_prompt(task: Dict[str, Any], condition: str) -> str:
    """Build the task message with buggy code and visible tests only."""
    return f"""TASK ID: {task['task_id']}
REPOSITORY: {task.get('repo', 'unknown')}
FILE: {task.get('target_file', 'unknown')}

BUGGY SOURCE CODE (complete file):
--- BEGIN BUGGY SOURCE ---
{task.get('mutated_code', '')}
--- END BUGGY SOURCE ---

TASK PROMPT:
{task['prompt']}

VISIBLE TEST (hidden oracle is withheld):
--- BEGIN VISIBLE TEST ---
{task.get('visible_test', '')}
--- END VISIBLE TEST ---

CONDITION: {condition}
"""

# ---------------------------------------------------------------------------
# Core async agent runner
# ---------------------------------------------------------------------------

async def run_agent_on_task(
    task: Dict[str, Any],
    model_name: str,
    condition: str = "A",
    temperature: float = 0.2,
    max_turns: int = 15,
) -> Dict[str, Any]:
    """
    Runs a Gemma 4 agent on a single SciSilentBench task and returns a result dict:
      {
        "task_id": str,
        "final_code": str,
        "decision": "SUBMIT" | "ABSTAIN",
        "confidence": float,
        "tokens": int,
        "wall_time_s": float,
      }
    """
    start_time = time.time()

    # --- Tools exposed to the agent ---
    read_file_tool = FunctionTool(read_file)
    run_pytest_tool = FunctionTool(run_pytest)
    run_python_tool = FunctionTool(run_python)

    def check_visible_test(final_code: str) -> str:
        """Run this task's visible test against a proposed complete source file."""
        try:
            passed = run_code_check(final_code, task["visible_test"])
            return "VISIBLE TEST PASSED" if passed else "VISIBLE TEST FAILED"
        except Exception as exc:
            return f"VISIBLE TEST ERROR: {type(exc).__name__}: {exc}"

    check_visible_tool = FunctionTool(check_visible_test)

    # We capture the submitted patch via a closure.
    submission: Dict[str, Any] = {"final_code": "", "decision": "ABSTAIN", "confidence": 0.0}

    def submit_patch(final_code: str, confidence: float = 0.5) -> str:
        """Called by the agent when it wants to submit its patch."""
        submission["final_code"] = final_code
        submission["decision"] = "SUBMIT"
        submission["confidence"] = float(confidence)
        return "Patch received. Agent may now terminate."

    def abstain(reason: str = "", confidence: float = 0.0) -> str:
        """Called by the agent when it chooses to abstain."""
        submission["final_code"] = ""
        submission["decision"] = "ABSTAIN"
        submission["confidence"] = float(confidence)
        submission["reason"] = reason
        return "Abstention recorded. Agent may now terminate."

    submit_tool = FunctionTool(submit_patch)
    abstain_tool = FunctionTool(abstain)

    # --- System prompt by condition ---
    system_prompt = (
        CONDITION_A_SYSTEM_PROMPT if condition == "A" else CONDITION_B_SYSTEM_PROMPT
    )

    # --- Build the agent ---
    agent = Agent(
        name=f"scisilentbench_agent_{condition}",
        model=_make_litellm_model(model_name),
        instruction=system_prompt,
        generate_content_config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=2048),
        tools=[
            read_file_tool,
            run_pytest_tool,
            run_python_tool,
            check_visible_tool,
            submit_tool,
            abstain_tool,
        ],
    )

    # --- Session + runner (ASYNC — must be awaited) ---
    app_name = "scisilentbench"
    user_id = "eval"
    session_id = f"session_{task['task_id']}_{condition}"
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    user_msg = types.Content(
        role="user",
        parts=[types.Part(text=_build_user_prompt(task, condition))],
    )

    # --- Run the agent turn loop ---
    tokens_used = 0
    turns = 0
    raw_text_parts = []
    tool_calls = []

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_msg,
        ):
            turns += 1
            content = getattr(event, "content", None)
            for part in getattr(content, "parts", []) or []:
                text_part = getattr(part, "text", None)
                if text_part:
                    raw_text_parts.append(text_part)
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    tool_calls.append({
                        "name": getattr(function_call, "name", ""),
                        "args": getattr(function_call, "args", {}),
                    })
            # Track approximate token usage if the event exposes it
            usage = getattr(event, "usage_metadata", None)
            if usage is not None:
                tokens_used += getattr(usage, "total_token_count", 0) or 0

            # Stop if the agent has submitted or abstained
            if submission["decision"] != "ABSTAIN" or turns >= max_turns:
                break
            if submission.get("reason") is not None and submission["decision"] == "ABSTAIN" and turns >= max_turns:
                break
    except Exception as e:
        # If the agent runtime throws, capture it and fall back to abstention.
        submission["decision"] = "ABSTAIN"
        submission["reason"] = f"agent runtime error: {type(e).__name__}: {e}"

    wall_time_s = time.time() - start_time

    return {
        "task_id": task["task_id"],
        "final_code": submission["final_code"],
        "decision": submission["decision"],
        "confidence": submission["confidence"],
        "reason": submission.get("reason", ""),
        "tokens": tokens_used,
        "wall_time_s": wall_time_s,
        "turns": turns,
        "raw_agent_output": "\n".join(raw_text_parts),
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Sync wrapper for notebooks / quick tests
# ---------------------------------------------------------------------------

def run_agent_on_task_sync(task, model_name, condition="A", temperature=0.2):
    """Synchronous convenience wrapper."""
    return asyncio.run(
        run_agent_on_task(
            task,
            model_name=model_name,
            condition=condition,
            temperature=temperature,
        )
    )


if __name__ == "__main__":
    # Quick smoke test
    demo_task = {
        "task_id": "SMOKE-001",
        "repo": "demo",
        "target_file": "demo.py",
        "prompt": "Fix the bug in demo.py.",
    }
    out = run_agent_on_task_sync(demo_task, model_name="gemini/gemini-1.5-flash")
    print(json.dumps(out, indent=2))