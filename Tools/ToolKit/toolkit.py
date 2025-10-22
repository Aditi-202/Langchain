from langchain_core import tool

@tool
def add(a, b) -> int:
    return a+b

@tool
def multiply(a, b) -> int:
    return a*b

class MathToolkit:
    def get_tools(self):
        return [add, multiply]
    
toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)
    