from langchain.chat_models import init_chat_model
from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.checkpoint.memory import InMemorySaver
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from uuid import uuid4
from typing import Optional, Callable, List, Dict, Any, Tuple
from enum import Enum, auto

from .agent_tools import *

class HumanEvalFixVersion(Enum):
    WITH_TESTS = auto()
    WITH_DOCSTRING = auto()


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
            max_new_tokens=1024,
            temperature=0.2,
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


    def build_prompt_msg(self, prompt: str = ""):
        content = prompt
        return {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }

    def build_agent_prompt_from_version(
        self,
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
        if version == HumanEvalFixVersion.WITH_TESTS:
            return (
                "You are a ReAct-style coding agent.\n\n"
                "You have two tools available:\n"
                "1) run_tests_sandbox(code, tests): execute your candidate function together with the tests in a restricted sandbox.\n"
                "2) finalize_solution(code): when you are done, call this to return ONLY the final fixed function code.\n\n"
                "Guidelines:\n"
                "- Keep your reasoning steps short and focused. Do not write long explanations.\n"
                "- Instead of thinking too much, frequently modify the code and run run_tests_sandbox.\n"
                "- Use an iterative approach: fix one issue, test, then refine.\n"
                "- Stop as soon as the tests pass, and call finalize_solution with the corrected function.\n\n"
                f"Here is a buggy function:\n{buggy_code}\n\n"
                f"Here are the unit tests:\n{test}\n\n"
                "Your task: fix the function so it passes all the tests. Follow the guidelines above."
            )

        if version == HumanEvalFixVersion.WITH_DOCSTRING:
            return (
                "You are a ReAct-style coding agent. You have two tools available: \n"
                "1) run_tests_sandbox(code, tests): execute your candidate function and the tests in a restricted sandbox.\n"
                "2) finalize_solution(code): when you are done, call this to return ONLY the final fixed function code.\n\n"
                f"Here is a buggy function with docstring:\n{buggy_code}\n"
                f"Your task: fix the function to implement the correct behavior. Use run_tests_sandbox as needed, then finish by calling finalize_solution with ONLY the fixed function code."
            )

        # Fallback
        return buggy_code or starting_prompt or ""

    def create_thread_config(
        self,
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
                print("lala")
                if should_stop and should_stop():
                    transcript_lines.append("[info] Stopping task on request.")
                    break
                if self.debug:
                    print("UPDATE:", update)
                for node, payload in update.items():
                    transcript_lines.append(f"[{node}] {payload}")
                    print(f"[{node}] {payload}")
        except Exception as e:
            transcript_lines.append(f"[error] {e}")
        return "\n".join(str(line) for line in transcript_lines)

    def start_task(
        self,
        prompt: str,
        tests: str = "",
        tests_visible_to_model: bool = False,
        max_steps: int = 20,
        should_stop: Optional[Callable[[], bool]] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Start a single task with a custom prompt and optional stopping callback."""
        input_prompt = self.build_prompt_msg(prompt)
        config, tid = self.create_thread_config(max_steps=max_steps, thread_id=thread_id)
        if self.debug:
            print(f"Starting task with thread_id={tid}")
        return self.stream_agent_updates(input_prompt, config, should_stop=should_stop)


    def provide_fixes_raw(self, prompt: str, tests: str, tests_visible_to_model: bool = False):
        """
        Run the ReAct agent on the given prompt and return a readable transcript
        that includes the reasoning (Thought) and action steps, as well as any
        tool results. Each call uses a fresh thread_id so no context is carried
        between tasks. The agent is expected to call finalize_solution when done.
        """
        return self.start_task(prompt, tests, tests_visible_to_model, max_steps=20)

    @staticmethod
    def extract_final_solution(transcript: str) -> str:
        """
        Extract the final solution code from the transcript produced by the agent.
        The finalize_solution tool returns a string beginning with "[Final Solution]".
        We return everything after that marker; if not found, return the original transcript.
        """
        if not transcript:
            return ""
        marker = "[Final Solution]"
        if marker in transcript:
            # take the last occurrence to prefer the final call
            idx = transcript.rfind(marker)
            return transcript[idx + len(marker):].strip()
        return transcript


    def run_tests_final_sandbox(self, code: str, tests: str) -> str:
        """
        Sandbox execution: runs code and tests in an isolated environment.
        Returns a summary of test results.
        """
        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "round": round,

            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "sorted": sorted,

            "all": all, "any": any, "print": print,

            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,

            "reversed": reversed, "divmod": divmod, "pow": pow,
        }
        safe_globals = {"__builtins__": safe_builtins}
        local_env = {}

        try:
            exec(code, safe_globals, local_env)
        except Exception as e:
            return f"[Error in function code] {e}"
        try:
            exec(tests, safe_globals, local_env)
            return "All tests passed"
        except AssertionError as e:
            return f"Test failed: {e}"
        except Exception as e:
            return f"Error while running tests: {e}"


    def process(self, version: HumanEvalFixVersion, buggy_code: str, test: str):
        """
        Process a buggy function according to HumanEvalFix variant.

        Args:
            version: Enum specifying which input variant to use.
            buggy_code: Buggy function code (with or without docstring).
            test: Test cases.

        Returns:
            Tuple[str, bool]: (fixed_code, passed_tests)
        """

        # agent gets the buggy code as input
        # it should use ReAct style to provide the fixed code
        # the steps of thinking should be visible
        # after, the OctoPack - testing
        # here the steps of agent's solution should be visible

        agent_input = self.build_agent_prompt_from_version(version=version, buggy_code=buggy_code, test=test)
        print("prompt built")
        transcript = self.provide_fixes_raw(
            agent_input,
            tests=test,
            tests_visible_to_model=(version == HumanEvalFixVersion.WITH_TESTS)
        )
        fixed_code_by_agent = self.extract_final_solution(transcript)

        result_run = self.run_tests_final_sandbox(fixed_code_by_agent, test)
        return fixed_code_by_agent, (result_run == "All tests passed")
