from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.checkpoint.memory import InMemorySaver
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from uuid import uuid4
from typing import Callable, Any, Tuple, Optional
from enum import Enum, auto

from .agent_tools import *

class HumanEvalFixVersion(Enum):
    WITH_TESTS = auto()
    WITH_DOCSTRING = auto()


def extract_final_code(transcript: dict | str) -> str:
    """
    Extract the finalized code from the agent's transcript/tool output.
    Robustly supports:
    - Direct tool dict output {action: finalize_solution, action_input: {code: ...}}
    - String output returned from finalize_solution tool: "[Final Solution]\n<code>"
    - Fenced code blocks ```python ... ``` inside the payload
    - JSON-like strings containing "action": "finalize_solution"
    Falls back to returning the stripped transcript.
    """
    import re

    # 1) Structured dict with finalize_solution
    if isinstance(transcript, dict) and transcript.get("action") == "finalize_solution":
        code = transcript.get("action_input", {}).get("code", "")
        if isinstance(code, str):
            # Prefer fenced block if present
            m = re.search(r"```(?:python)?\n(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
            return (m.group(1) if m else code).strip()
        return ""

    # 2) String outputs
    if isinstance(transcript, str):
        text = transcript.strip()

        # 2a) Output from finalize_solution tool wrapper
        if text.startswith("[Final Solution]"):
            # Remove the header and keep the rest
            text = text.split("\n", 1)[1] if "\n" in text else ""

        # 2b) Try fenced code first
        m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # 2c) JSON-like log with action finalize_solution and a "code" field
        if '"action"' in text and 'finalize_solution' in text and '"code"' in text:
            # Try to find triple-quoted or single-line code value
            triple = re.search(r'"code"\s*:\s*"""(.*?)"""', text, flags=re.DOTALL)
            if triple:
                return triple.group(1).strip()
            single = re.search(r'"code"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
            if single:
                # Unescape common sequences
                s = single.group(1)
                s = s.encode('utf-8').decode('unicode_escape')
                return s.strip()

        # 2d) As a last resort, if there's a def ...(:) block, grab from first def to end
        def_idx = text.find("def ")
        if def_idx != -1:
            return text[def_idx:].strip()

        return text

    # Fallback
    return str(transcript).strip()


def create_thread_config(
        max_steps: int = 20,
        thread_id: Optional[str] = None,
) -> Tuple[RunnableConfig, str]:
    """
    Create a RunnableConfig with a new or provided thread_id. Returns (config, thread_id).
    """
    if thread_id is None:
        thread_id = str(uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "max_steps": max_steps}}
    return config, thread_id


def build_agent_prompt_from_version(
        version: Optional["HumanEvalFixVersion"],
        buggy_code: Optional[str] = None,
        test: Optional[str] = None,
        starting_prompt: Optional[str] = None,
) -> str:
    """
    Build the starting prompt for the agent from a version enum or a custom starting prompt.
    If starting_prompt is provided, it takes precedence over version-based templates.
    """
    if starting_prompt:
        return starting_prompt

    safe_buggy_code = buggy_code.replace("{", "{{").replace("}", "}}")
    safe_tests = test.replace("{", "{{").replace("}", "}}")

    prompt_tests = build_agent_prompt_docstring_tests(
            buggy_code_=safe_buggy_code,
            tests_=safe_tests,
        )
    print("\nPROMPT\n")
    print(prompt_tests)
    print("\nPROMPT\n")
    if version == HumanEvalFixVersion.WITH_TESTS:
        return prompt_tests

    # Fallback
    return buggy_code or starting_prompt or ""


def build_prompt_msg(prompt: str = ""):
    content = prompt
    return {
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ]
    }


def run_tests_final_sandbox(code: str, tests: str) -> str:
    """
    Sandbox execution: runs code and tests in an isolated environment.
    Accepts optional fenced code blocks and strips them before execution.
    Returns a summary of test results.
    """
    import re

    def strip_fence(s: str) -> str:
        if not isinstance(s, str):
            return ""
        m = re.search(r"```(?:python)?\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        return (m.group(1) if m else s).strip()

    code = strip_fence(code)
    tests = strip_fence(tests)

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "round": round,

        "len": len, "range": range, "enumerate": enumerate, "zip": zip,
        "sorted": sorted,

        "all": all, "any": any, "print": print,

        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,

        "reversed": reversed, "divmod": divmod, "pow": pow,
    }
    my_safe_globals = {"__builtins__": safe_builtins}
    local_env = {}

    try:
        exec(code, my_safe_globals, local_env)
    except Exception as e:
        return f"[Error in function code] {e}"
    try:
        exec(tests, my_safe_globals, local_env)
        return "All tests passed"
    except AssertionError as e:
        return f"Test failed: {e}"
    except Exception as e:
        return f"Error while running tests: {e}"


class Agent :
    """ minimal version of agent to fix bugs in code"""
    def __init__(self, model_name="Qwen/Qwen3-0.6B", debug=False, tools: Optional[List[Any]] = None):
        # Keep init lightweight to avoid long blocking during construction
        self.memory = InMemorySaver()
        self.debug = debug
        self.model_name = model_name
        self.llm = None
        self.agent = None
        self._ensure_initialized(tools=tools)

    def _ensure_initialized(self, tools: Optional[List[Any]] = None):
        """Lazily initialize the local model and ReAct agent on first use."""
        if self.agent is not None:
            return
        if self.debug:
            print(f"Initializing model and agent with model_name={self.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map=None
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=5000,
            temperature=0.6,
            top_p = 0.95,
            top_k=20,
        )
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        self.llm = ChatHuggingFace(llm=hf_llm)
        if tools is None:
            tools = []
        # Create the ReAct agent only after LLM is ready
        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            checkpointer=self.memory,
        )

    def stream_agent_updates(
        self,
        input_prompt: Dict[str, Any],
        config: RunnableConfig,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> str:
        """
        Stream updates from the agent and build a transcript. If should_stop() returns True,
        the streaming loop will end early and the transcript will note it.
        """
        transcript_lines: List[str] = []
        print("agent updates")
        try:
            for update in self.agent.stream(
                Command(update=input_prompt),
                config=config,
                stream_mode="updates",
            ):
                print("inside updates")
                if should_stop and should_stop():
                    transcript_lines.append("[info] Stopping task on request.")
                    break
                if self.debug:
                    print("UPDATE:", update)
                for node, payload in update.items():
                    transcript_lines.append(f"[{node}] {payload}")
                    print(f"[{node}] {payload}\n")
                    if node == "agent":
                        print(f"[Agent Thought] {payload}")
                    elif node == "tools":
                        print(f"[Tool Result] {payload}")
                    if node == "tools":
                        if payload.get("name") == "finalize_solution":
                            final_code = payload["output"]
                            print(f"{final_code=}")
                            print("\nGot Final Code\n")
        except Exception as e:
            transcript_lines.append(f"[error] {e}")
        return "\n".join(str(line) for line in transcript_lines)

    def start_task(
        self,
        prompt: str,
        max_steps: int = 20,
        should_stop: Optional[Callable[[], bool]] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Start a single task with a custom prompt and optional stopping callback."""
        input_prompt = build_prompt_msg(prompt)
        config, tid = create_thread_config(max_steps=max_steps, thread_id=thread_id)
        if self.debug:
            print(f"Starting task with thread_id={tid}")
        return self.stream_agent_updates(input_prompt, config, should_stop=should_stop)


    def provide_fixes_raw(self, prompt: str):
        """
        Run the ReAct agent on the given prompt and return a readable transcript
        that includes the reasoning (Thought) and action steps, as well as any
        tool results. Each call uses a fresh thread_id so no context is carried
        between tasks. The agent is expected to call finalize_solution when done.
        """
        return self.start_task(prompt, max_steps=20)

    def process(self, task_id: str, version: HumanEvalFixVersion, buggy_code: str, test: str):
        """
        Process a buggy function according to HumanEvalFix variant.

        Args:
            task_id: id of the task instance
            version: Enum specifying which input variant to use.
            buggy_code: Buggy function code (with or without docstring).
            test: Test cases.

        Returns:
            Tuple[str, bool]: (fixed_code, passed_tests)
        """

        agent_input = build_agent_prompt_from_version(version=version, buggy_code=buggy_code, test=test)
        transcript = self.provide_fixes_raw(agent_input)
        fixed_code_by_agent = transcript
        with open("all_runs_3.txt", "a") as f:
            f.write(f"{task_id=}\n{buggy_code=}\n{fixed_code_by_agent=}\n\n-=*=- -=*=- -=*=- -=*=- -=*=-\n\n")
        print(f"\n\n{buggy_code = }\n\n")
        print(f"\n{fixed_code_by_agent=}\n\n")
        fixed_code_by_agent = extract_final_code(transcript)
        print(f"\n\n{fixed_code_by_agent=}\n\n")
        # Try OctoPack tests runner if configuration and tests exist; otherwise fallback to sandbox tests
        result_run: str
        try:
            from evaluation.octopack_runner import run_octopack_tests  # local import to avoid hard dependency
            import os
            has_octopack = os.path.exists(os.path.join(os.getcwd(), "octopack.yaml"))
            if has_octopack:
                passed, details = run_octopack_tests(fixed_code_by_agent, root_dir=os.getcwd())
                print("[OctoPack] Details:\n" + "\n".join(details))
                result_run = "All tests passed" if passed else "OctoPack tests failed"
            else:
                result_run = run_tests_final_sandbox(fixed_code_by_agent, test)
        except Exception as e:
            print(f"[OctoPack] Runner error: {e}. Falling back to sandbox.")
            result_run = run_tests_final_sandbox(fixed_code_by_agent, test)
        print(f"\n\n{result_run}\n")
        return fixed_code_by_agent, (result_run == "All tests passed")
