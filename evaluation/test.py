from datasets import load_dataset
import pandas as pd
from agent.agent import Agent
from agent.agent import HumanEvalFixVersion
from agent.agent_tools import *

# Get dataset for agent evaluation - 'test' represents HumanEvalFix
data = load_dataset("bigcode/humanevalpack", "python", split="test")
df = pd.DataFrame(data)

my_agent = Agent(tools=agent_tools, debug=False)

for ind, test_item in df.iterrows():
    print(f"{test_item['task_id']=}")
    # testing on the buggy solution (with docstring),
    # the agent has tests
    declaration = test_item["prompt"] # test_item["declaration"] # providing more info than just declaration
    # from the paper: incorrect function body with a bug
    buggy_solution = test_item["buggy_solution"]
    buggy_code = declaration + buggy_solution
    # in at least one testcase buggy solution fails
    test = test_item["test"]
    code_t, success_t = my_agent.process(
        task_id=test_item["task_id"],
        version=HumanEvalFixVersion.WITH_TESTS,
        buggy_code=buggy_code,
        test=test
    )
    print("--======--" * 5)
    print(test_item["task_id"])
    print(success_t)
    print(code_t, success_t)
    print("\n\fFINISHED TASK\n\n")