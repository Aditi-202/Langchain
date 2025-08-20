from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()
model = ChatOpenAI()

st.header('Research Tool')

Paper_input = st.selectbox("select research paper name",["select..","attention is all you need","bert:pre-training of deep bidirectional transformation","gpt-3 langauage are few learners","diffusion models beat GANs on image synthesis"])

lenght_input = st.selectbox("seleect explaination lenght",["short(1-2 paragraph)", "medium(3-5 paragraph)","long(detailed explaination)"])

#template
template = PromptTemplate(
    template=""" ___ enter your template here ___""",
    input_variables=['Paper_input','lenght_input']
)

#fill the placeholder
prompt = template.invoke({
    'Paper_input' : Paper_input,
    'lenght_input' : lenght_input
})


if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)