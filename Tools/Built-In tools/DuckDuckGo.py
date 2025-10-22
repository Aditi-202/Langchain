from langchain_community.tools import DuckDuckGoSearchRun

search_tools = DuckDuckGoSearchRun()
results = search_tools.invoke(' top news in Artifical intelligence news')

print(results)