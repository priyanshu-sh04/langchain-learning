from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Student (BaseModel):
    
    name: str
    roll_no: int = 21  # set default value
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        gt=0,
        lt=10,
        description="A decimal value representing the cgpa of students"
    )
    

new_student = {"name":"priyanshhu", "email":"abc@gmail.com", "cgpa": 8.4, "age":21}

student = Student(**new_student)


print(dict(student))
