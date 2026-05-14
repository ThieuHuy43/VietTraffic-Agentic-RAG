from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv()
try:
    wrapper = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    search = TavilySearchResults(api_wrapper=wrapper, max_results=1)
    print("Success wrapper")
except Exception as e:
    print(e)
