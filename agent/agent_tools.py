from langchain_core.tools import tool, Tool

from RestrictedPython import compile_restricted, safe_globals
from datetime import datetime
from typing import List, Dict, Optional, Union

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
    Returns:
        {
            "function_status": "ok" or "incorrect function",
            "test_status": "all tests passed" or "not all tests passed",
            "violations": list of forbidden function attempts
        }
    """
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

