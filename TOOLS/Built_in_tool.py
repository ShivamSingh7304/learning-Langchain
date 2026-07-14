from langchain_community.tools import DuckDuckGoSearchRun

search_res= DuckDuckGoSearchRun()
result = search_res.invoke("fifa news")

#print(result)


print("\n" + "=" * 80)
print("Search Results")
print("=" * 80)
print(result)
print("=" * 80)


