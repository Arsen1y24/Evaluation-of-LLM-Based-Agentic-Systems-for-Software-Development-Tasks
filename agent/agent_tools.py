from langchain_core.tools import tool, Tool

from RestrictedPython import compile_restricted, safe_globals
from datetime import datetime
from typing import List, Dict, Union

LOG_FILE = "sandbox.log"

def log_message(message: str):
    """Log message to file with timestamp"""
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} - {message}\n")

class SafeLocals(dict):
    """
    Local namespace for sandboxed code.
    Intercepts access to forbidden names and returns a dummy function.
    Logs all such attempts.
    """
    def __getitem__(self, key):
        if key not in self:
            log_message(f"[WARNING] Attempted to use forbidden name '{key}'")
            def dummy(*args, **kwargs):
                log_message(f"[DUMMY] Call to '{key}' ignored")
                return None
            return dummy
        return super().__getitem__(key)

@tool
def run_tests_sandbox(code: str, tests: str) -> Dict[str, Union[str, List[str]]]:
    """
    Executes agent code and tests in a restricted sandbox.
    Accepts code and tests optionally wrapped in fenced blocks (```python ... ```).
    Returns:
        {
            "function_status": "ok" or "incorrect function",
            "test_status": "all tests passed" or "not all tests passed",
            "violations": list of forbidden function attempts
        }
    """
    import re
    print("\n\nrunning tests!!!!!!!!!!!!  ! ! ! !  !  ! \n\n")

    # Reset log file for a fresh run
    with open(LOG_FILE, "w") as _f:
        _f.write("")

    def strip_fence(s: str) -> str:
        if not isinstance(s, str):
            return ""
        m = re.search(r"```(?:python)?\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        return (m.group(1) if m else s).strip()

    code = strip_fence(code)
    tests = strip_fence(tests)

    sandbox_globals = safe_globals.copy()
    # Add safe built-in functions
    sandbox_globals.update({
        "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
        "len": len, "range": range, "enumerate": enumerate, "zip": zip,
        "sorted": sorted, "all": all, "any": any, "print": print,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "reversed": reversed, "divmod": divmod, "pow": pow,
    })

    safe_locals = SafeLocals()
    result: Dict[str, Union[str, List[str]]] = {
        "function_status": "",
        "test_status": "",
        "violations": []
    }

    # Execute function code
    try:
        byte_code_func = compile_restricted(code, "<string>", "exec")
        exec(byte_code_func, sandbox_globals, safe_locals)
        result["function_status"] = "ok"
    except Exception as e:
        log_message(f"[ERROR] Function code error: {e}")
        result["function_status"] = "incorrect function"

    # Execute test code
    try:
        byte_code_tests = compile_restricted(tests, "<string>", "exec")
        exec(byte_code_tests, sandbox_globals, safe_locals)
        if result["function_status"] == "ok":
            result["test_status"] = "all tests passed"
        else:
            result["test_status"] = "not all tests passed"
    except AssertionError as e:
        log_message(f"[TEST FAIL] {e}")
        result["test_status"] = "not all tests passed"
    except Exception as e:
        log_message(f"[ERROR] Test code error: {e}")
        result["test_status"] = "incorrect function"

    # Collect all violations from log
    with open(LOG_FILE, "r") as f:
        for line in f:
            if "[WARNING]" in line or "[DUMMY]" in line:
                result["violations"].append(line.strip())

    return result


@tool
def finalize_solution(code: str) -> str:
    """
    Finishes the task and returns the final solution.
    Agent can use this tool to finish the task execution.
    """
    return f"[Final Solution]\n{code}"




agent_tools = [
    Tool(
        name="run_tests_sandbox",
        func=run_tests_sandbox,
        description="Executes the generated code and tests in a restricted sandbox."
    ),
    Tool(
        name="finalize_solution",
        func=finalize_solution,
        description="Returns the final solution code after agent completes all steps."
    )
]

def build_agent_prompt_docstring_tests(buggy_code_: str, tests_: str) -> str:
    return f"""
You are an AI code fixer specialized in Python. Your goal is to repair the buggy function below so that it passes all provided tests.

Buggy code:
```python
{buggy_code_}
```

Tests (for your reference; do not modify):
```python
{tests_}
```

Critical instructions:

1) Strict output format (JSON only): For every step you must output a single JSON object with the following keys exactly:
{{
  "Thought": "<your brief reasoning>",
  "action": "run_tests_sandbox" | "finalize_solution",
  "action_input": {{ ... }}
}}
Do not output any text outside this JSON object. No markdown outside JSON.

2) Code field formatting: Whenever you send code in action_input.code, it MUST be wrapped in a fenced code block using triple backticks and the language tag python, and it MUST contain only the complete fixed function (and any minimal helpers it directly needs). Example:
"code": "```python\n<only the function code>\n```"
No prose, no explanations, no test code. Returning prose will cause a parsing error.

3) Tool order: First iterate by calling run_tests_sandbox with your current attempt. Only after all tests pass, call finalize_solution with the same code block.

4) Consistency: Keep the original function signature and name. Do not rename parameters or change the API.

Available tools and IO formats:

run_tests_sandbox
- Purpose: Execute your current code against the tests.
- Input JSON schema:
  {{
    "code": "```python\n<current code>\n```",
    "tests": "```python\n<tests code>\n```"
  }}
- Output: Indicates whether tests pass and any violations.

finalize_solution
- Purpose: Return the final fixed code when all tests pass.
- Input JSON schema:
  {{
    "code": "```python\n<final fixed code>\n```"
  }}
- After this tool call, stop.

Required JSON format for each action (copy exactly, replacing placeholders):
{{
  "Thought": "<your reasoning about the current step>",
  "action": "run_tests_sandbox",
  "action_input": {{
    "code": "```python\n<current fixed code>\n```",
    "tests": "```python\n{tests_}\n```"
  }}
}}

When all tests pass, respond with:
{{
  "Thought": "All tests have passed. I will finalize the fixed function.",
  "action": "finalize_solution",
  "action_input": {{
    "code": "```python\n<final fixed code>\n```"
  }}
}}

Begin fixing the function now. Follow the reasoning -> tool call cycle strictly.
"""