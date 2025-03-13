import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
# from tools.tools import get_profile_url_tavily
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def get_profile_url_tavily(name: str):
    """Searches for Linkedin or Twitter Profile Page."""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res[0]["url"]


def lookup(name: str) -> str:
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile"
    )
    
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                              Your answer should contain only a URL"""
                              
    promtemplate = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    
    tools_for_agent =[
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]
    
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agentExecutor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    
    result = agentExecutor.invoke(
        input={"input": promtemplate.format_prompt(name_of_person=name)}
    )
    
    linked_profile_url = result["output"]
    return linked_profile_url
    
    

def main():
    print(lookup(name="Eden Marco Udemy"))

if __name__=="__main__":
    main()