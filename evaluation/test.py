from datasets import load_dataset
import pandas as pd
from agent.agent import Agent
import agent.agent as agent
from agent.agent import HumanEvalFixVersion

# Get dataset for agent evaluation - 'test' represents HumanEvalFix
data = load_dataset("bigcode/humanevalpack", "python", split="test")
df = pd.DataFrame(data)

my_agent = Agent()


for ind, test_item in df.iterrows():
    # testing on the buggy solution (without docstring),
    # the agent has tests
    declaration = test_item["declaration"]
    # from the paper: incorrect function body with a bug
    buggy_solution = test_item["buggy_solution"]
    buggy_code = declaration + buggy_solution
    # in at least one testcase buggy solution fails
    test = test_item["test"]
    my_agent.process(
        version=HumanEvalFixVersion.WITH_TESTS,
        buggy_code=buggy_solution,
        test=test
    )


    # context + declaration of the function with docstring and
    # incorrect implementation of a function with a bug
    buggy_impl_with_docstring = test_item["prompt"] + test_item["buggy_solution"]
    my_agent.process(
        version=HumanEvalFixVersion.WITH_DOCSTRING,
        buggy_code=buggy_impl_with_docstring,
        test=test
    )