# install langchain_experimental
from langchain_community import ShellTool

shell = ShellTool()
results = shell.invoke(' whoami')

print(results)