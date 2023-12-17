import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.load import dumpd, load
import json

def _create_vectorstore(file_path: str) -> DocArrayInMemorySearch:
    document = TextLoader(file_path, encoding="utf8").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document)

    embedding = OpenAIEmbeddings()

    vectorstore = DocArrayInMemorySearch.from_documents(docs, embedding)
    return vectorstore

vectorstore = _create_vectorstore(file_path="data/rules.txt")

# if not os.path.isfile('vectorscore.json'):
#     vectorstore = _create_vectorstore(file_path="data/rules.txt")
#     vectorstore_dict = dumpd(vectorstore)
#     with open('vectorscore.json', 'w') as f:
#         json.dump(vectorstore_dict, f)
# else:
#     with open('vectorscore.json', 'r') as f:
#         vectorstore = load(json.load(f))


retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Refuse to answer questions that are not related to magic the gathering/MTG

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

print("Enter your MTG questions:")
user_input = input()
output = chain.invoke(user_input)
print(output)
