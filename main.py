import json
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

# with open("./UEFA_Euro_2020_Data/Euro2020.json") as f:
#    data = {"data": json.load(f)}

with open("./oi.json") as f:
    data = json.load(f)

splitter = RecursiveJsonSplitter(max_chunk_size=200)

docs = splitter.create_documents(texts=[data])

model_local = ChatOllama(model="llama3")

vectorstore = Chroma.from_documents(
    documents=docs,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)

retriever = vectorstore.as_retriever()

print("Before RAG\n")
before_rag_template = "Who was the winner of the 2023 {topic} tournament?"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Champions League"}))

print("After RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("Who was the winner of the Champions League in 2023??"))
