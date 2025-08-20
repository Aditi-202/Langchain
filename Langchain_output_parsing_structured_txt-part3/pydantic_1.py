from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str
    age: Optional[int] = None # if value is there it gets printed else it is none
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)# sets like a range greater than & lesser than
    sgpa: float = Field(gt=0, lt=10, default= 9)# if we don't mention sgpa in object, it still prits the sgpa in console because of default value

new_student = {'name': 'aditi', 'email': 'abc@gmail.com', 'cgpa': 5}

student = Student(**new_student)

print(student)