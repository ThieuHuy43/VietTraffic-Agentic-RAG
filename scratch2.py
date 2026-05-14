from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import inspect

print("Methods in wrapper:")
print([m for m in dir(TavilySearchAPIWrapper) if not m.startswith("_")])
