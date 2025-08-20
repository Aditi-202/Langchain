from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):

    name: str
    age: Optional[int] = None # if value is there it gets printed else it is none

new_student = {'name': 'aditi'}

student = Student(**new_student)

print(student)