from dotenv import load_dotenv
from langchain.agents import tool, Tool
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_groq import ChatGroq
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str

from typing import Union

from callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """
    Calculate the length of the given text in characters.
    """
    # print(f"get_text_length enter with {text=}")
    # text = text.split("\n").strip('"')
    return len(text)

def find_tool_by_name(tools: list[Tool], tool_name:str)->Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool}")
            

if __name__=="__main__":
    print("Hello ReAct LangChain")
    # print(get_text_length.invoke(input={'text':'Dog'}))
    tools = [get_text_length]
    
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    
    llm = ChatGroq(temperature=0, 
                   stop=["\nObservation", "Observation"],
                   callbacks=[AgentCallbackHandler()],
                   model_name="mixtral-8x7b-32768")
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad" : lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    
    intermediate_steps = []
    
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)
    
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        
        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))
        
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the word: DOG",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)
    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)