from langchain.chat_models import init_chat_model
from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langchain_core.tools import tool
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline


class Agent :
    """ minimal version of agent to fix bugs in code"""
    def __init__(self, model_name="Qwen/Qwen3-0.6B", debug=False):
        self.memory = InMemorySaver()
        self.debug = debug

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
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


        @tool
        def code_interpreter(code: str):
            """interprets code"""
            if self.debug:
                print("Executing code ", code)
            return "result"


        self.tools = [
                {
                    "name": "code_interpreter",
                    "func": code_interpreter,
                    "description": "execute Python code safely in sandbox"
                }
        ]

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory
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

    def provide_fixes_raw(self, prompt: str):
        input_prompt = self.build_prompt_msg(prompt)
        if self.debug:
            print("debugging")
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}
        response = self.agent.invoke(
            Command(
                update=input_prompt,
            ),
            config = config
        )
        return response
