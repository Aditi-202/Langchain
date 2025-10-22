from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="the first number to add")
    b: int = Field(required=True, description="the second number to add")

class multiplytool(BaseTool):
    name: str = 'multiply'
    description: str = "multiply 2 numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b:int) -> int:
        return a*b
    
multiply_tool = multiplytool()
result = multiply_tool.invoke({"a":3, "b":4})
print(result)
