from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

video_id = "i40oXOjETAM"  # only the id

try:
    # Try English first
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
except (NoTranscriptFound, NoTranscriptAvailable):
    # Fall back to the first available transcript (auto-generated or other language)
    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    first_transcript = transcripts.find_transcript([t.language_code for t in transcripts])
    transcript_list = first_transcript.fetch()
except TranscriptsDisabled:
    print("Transcript is disabled for this video.")
    transcript_list = []

# Flatten to plain text
if transcript_list:
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript[:500])  # Print only first few chars for readability
else:
    print("No transcript available.")
# got all the transcript not

#step1- text spliter(Indexing)
spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = spliter.create_documents([transcript])

len(chunks)

# embedding generation and storing in vector stor
embedding = HuggingFaceEmbeddings(model="model_name")
vector_store = FAISS.from_documents(chunks, embedding)

# step 2: retriever
retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k":4})

#retriever.invoke("what is RAG")

# step 3: Augumentation
prompt = PromptTemplate(
    template= """
            You are hellpful assistant.
            Answer ONLY from provided transcript context.
            If the context is insufficient, just say you don't know
            
            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
)

question = "any question"
retrieved_docs = retriever.invoke(question)

context_text = "\n \n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context': context_text, "question": question })

#step 4: Generation
answer = llm.invoke(final_prompt)
print(answer.content)

# building a chain
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke("add_your_question")
parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser

main_chain.invoke("add_you_question_regarding_video")