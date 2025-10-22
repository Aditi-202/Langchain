from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    input_variables=['question', 'text'],
    template="anser the following question \n {question} from the following text \n {text}"
)

parser = StrOutputParser()

loader = WebBaseLoader('https://www.amazon.in/Yash-Gallery-Womens-Floral-Anarkali/dp/B08SK12H2Q/ref=sxin_19_pa_sp_phone_search_thematic_sspa?content-id=amzn1.sym.4443ee66-42b9-40e8-893c-1284ba030b11%3Aamzn1.sym.4443ee66-42b9-40e8-893c-1284ba030b11&crid=KCBHRN4BER0Y&cv_ct_cx=a%2Bline%2Bkurti%2Bfor%2Bwomen&keywords=a%2Bline%2Bkurti%2Bfor%2Bwomen&pd_rd_i=B08SK27Z7P&pd_rd_r=8b57997d-185d-4256-bc15-6f3447a89d3d&pd_rd_w=lBHWP&pd_rd_wg=Iub8d&pf_rd_p=4443ee66-42b9-40e8-893c-1284ba030b11&pf_rd_r=RRK043NRT2XV1VSEA4RH&qid=1755259990&sbo=RZvfv%2F%2FHxDF%2BO5021pAnSA%3D%3D&sprefix=a%2Bline%2Bkurti%2Caps%2C335&sr=1-42-2ec22325-1003-449d-8aaf-c0bcc24717ae-spons&xpid=pscOVYB0NjxsM&sp_csd=d2lkZ2V0TmFtZT1zcF9waG9uZV9zZWFyY2hfdGhlbWF0aWM&th=1&psc=1')

docs = loader.load()

chain = prompt1 | model |parser
print(chain.invoke({'question' : 'what is the product we are talking about?', 'text': docs[0].page_content}))