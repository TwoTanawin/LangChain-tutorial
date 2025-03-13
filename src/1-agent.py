from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from agents.linkedin_lookup_agent import lookup
from third_parties.linkedin import scrape_linkedin_profile
# import os
# import getpass

def ice_break_with(name: str) -> str:
    linkedin_username = lookup(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)
    
    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    
    chain = summary_prompt_template | llm | StrOutputParser()
    
    res = chain.invoke(input={"information": linkedin_data})
    
    print(res)

def main():
    
    load_dotenv()
    ice_break_with(name="Eden Marco")
    
    


if __name__=="__main__":
    print("Hello LangChain")
    main()