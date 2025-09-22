from langchain.chat_models import init_chat_model
from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from uuid import uuid4

from enum import Enum, auto
class HumanEvalFixVersion(Enum):
    WITH_TESTS = auto()
    WITH_DOCSTRING = auto()


class Agent :
    """ minimal version of agent to fix bugs in code"""
    def __init__(self, model_name="Qwen/Qwen3-0.6B", debug=False):
        # Keep init lightweight to avoid long blocking during construction
        self.memory = InMemorySaver()
        self.debug = debug
        self.model_name = model_name
        self.llm = None
        self.agent = None


    def _ensure_initialized(self):
        """Lazily initialize the local model and ReAct agent on first use."""
        if self.agent is not None:
            return
        if self.debug:
            print(f"Initializing model and agent with model_name={self.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            device_map=None
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0,
        )
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        self.llm = ChatHuggingFace(llm=hf_llm)
        # Create the ReAct agent only after LLM is ready
        self.agent = create_react_agent(
            model=self.llm,
            tools=[],
            # tools=self.tools,
            checkpointer=self.memory,
            # max_steps=5,
            # allow_thoughts_without_action=True,
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


    def run_tests_sandbox(self, code: str, tests: str) -> str:
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


    def provide_fixes_raw(self, prompt: str):
        """
        Run the ReAct agent on the given prompt and return a readable transcript
        that includes the reasoning (Thought) and action steps, as well as any
        tool results. Each call uses a fresh thread_id so no context is carried
        between tasks.
        """
        # Ensure heavy components are initialized only when needed
        self._ensure_initialized()
        input_prompt = self.build_prompt_msg(prompt)

        # Fresh thread id per call to avoid carrying context between tasks
        thread_id = str(uuid4())

        if self.debug:
            print(f"Starting task with thread_id={thread_id}")

        config: RunnableConfig = {"configurable": {"thread_id": thread_id, "max_steps": 20}}
        transcript_lines = []

        try:
            # Stream updates so we can expose Thought/Action/Observation steps
            for update in self.agent.stream(
                    Command(update=input_prompt),
                    config=config,
                    stream_mode="updates"
            ):
                if self.debug:
                    print("UPDATE:", update)
                # update is a dict of node_name -> payload; stringify for readability
                for node, payload in update.items():
                    transcript_lines.append(f"[{node}] {payload}")
                    print(f"[{node}] {payload}")
        except Exception as e:
            transcript_lines.append(f"[error] {e}")

        # Return the transcript as a single string so the caller can see steps
        return "\n".join(str(line) for line in transcript_lines)


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

        agent_input = ""
        if version == HumanEvalFixVersion.WITH_DOCSTRING:
            agent_input = f"Here is a buggy function with docstring:\n{buggy_code}\nPlease fix the function so it would implement the correct behavior."
        elif version == HumanEvalFixVersion.WITH_TESTS:
            agent_input = f"Here is a buggy function:\n{buggy_code}\nHere are the unit tests:\n{test}\nPlease fix the function so it passes the tests."

        transcript = self.provide_fixes_raw(agent_input)

        return "resulting_code", "passed_tests"
