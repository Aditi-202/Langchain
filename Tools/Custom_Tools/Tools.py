from langchain_core import tool

#step1 creata a func
def multiply(a, b):
    """Multiply 2 numbers"""
    return a * b

#step2 add type hints
def multiply(a:int, b:int):
    """Multiply 2 numbers"""
    return a * b

#step3 add tool decorator
@tool # adding tool decortor, so that llm can interact direct with this
def multiply(a: int, b:int):
    """Multiply 2 numbers"""
    return a * b

result = multiply.invoke({"a":3, "b":5})
print(result)