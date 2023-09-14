from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import re
import sys
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory
import os
import re
os.environ['OPENAI_API_KEY'] = 'put-api-key-here'
llm_resto = ChatOpenAI(model="gpt-4")
prompt_template_resto = PromptTemplate(
    input_variables=['os', 'brand', 'usage', 'budget', 'size', 'additional'],
    template="phone Recommendation System:\n"
             "You are a phone recommender, As a phone recommender, your task is to assist users in discovering the perfect mobile device based on their unique preferences, budget, and needs. Your response should consist first of the brand name and model of up to 10 recommended phones, formatted as 1 - phone A description of phone configure 2 - phone B description of phone configure 3 - phone C description and explanation and then an explaination why? and order by cheapest price.\n"
             "preferred operating system: {os}\n"
             "preferred phone brand: {brand}\n"
             "usage of phone: {usage}\n"
             "budget: {budget}\n"
             "phone size: {size}\n"
             "additional information: {additional}\n."
)
PERSIST = False
query = None
if len(sys.argv) > 1:
    query = sys.argv[1]
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader("./data/", glob="./*.csv", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])
class Answer(BaseModel):
    os: str
    brand: str | None = None
    usage: str
    budget: str
    size: str
    additional: str
app = FastAPI()
@app.post("/Answer/")
async def answer(answer: Answer):
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1})
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    chain_resto = ConversationalRetrievalChain.from_llm(llm=llm_resto, retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}))
    input_data = {'os': answer.os,
                  'brand': answer.brand,
                  'usage': answer.usage,
                  'budget': answer.budget,
                  'size': answer.size,
                  'additional': answer.additional}
    # results = chain_resto.run(input_data)
    question = prompt_template_resto.format(os = answer.os, brand = answer.brand, usage = answer.usage, budget = answer.budget, size = answer.size, additional = answer.additional)
    result = chain_resto({"question": question, "chat_history": []})
    return {"result": result['answer']}