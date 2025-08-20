from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/falcon-7b-instruct",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm = llm)

result= model.invoke("what is the capital of India?")

print(result.content)
# this code is taking too much time to download & display because i don't have gpu so don't run this code with this model