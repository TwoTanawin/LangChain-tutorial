from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    information = """
Mark Elliot Zuckerberg (/ˈzʌkərbɜːrɡ/; born May 14, 1984) is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy.

Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the world's youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the world's wealthiest individuals. According to Forbes, as of March 2025, Zuckerberg's estimated net worth stood at US$214.1 billion, making him the second richest individual in the world,[2] behind Elon Musk and before Jeff Bezos.

Zuckerberg has used his funds to organize multiple large donations, including the establishment of the Chan Zuckerberg Initiative. A film depicting Zuckerberg's early career, legal troubles and initial success with Facebook, The Social Network, was released in 2010 and won multiple Academy Awards. His prominence and fast rise in the technology industry has prompted political and legal attention.
    """
    
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    
    llm = OllamaLLM(temperature=0, model="gemma3", host="localhost", port=11434)
    
    chain = summary_prompt_template | llm | StrOutputParser()
    
    res = chain.invoke(input={"information": information})
    
    print(res)
    
if __name__=="__main__":
    print("Hello LangChain")
    main()
