from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field # literal tells that it should be either this or that

load_dotenv()

model = ChatOpenAI()

#schema
#write a json_scheme
json_schema ={
    "title":"review",
    "type":"object",
    "properties":{
        "key_themes":{
            "type": "array",
            "items":{
                "type":"string"
            },
            "description" : "write all the key themes discussed in the review in the list"
        },
        "summary" : {
            "type": "string",
            "description" : "write all the key themes discussed in the review in the list"
        },
        "sentiment": {
            "type" : "string",
            "enum":["pos","neg"],
            "description" : "write all the key themes discussed in the review in the list"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description" : "write all the key themes discussed in the review in the list"
        },

        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description" : "write all the key themes discussed in the review in the list"
        },
        "name":{
            "type": ["string", "null"],
            "description": "write the name"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
        }

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke(""" the hardware is great, but the software feels bloated. there are too many pre installed apps that i can't remove. also the ui looks outdated compared to other brands. hoping for a software update to fix this.""")

print(result)