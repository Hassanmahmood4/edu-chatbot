import streamlit as st

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# ---------- CONFIG ----------
DATA_PATH = "data/knowledge.txt"
DB_PATH = "chroma_db"
MODEL_NAME = "llama3"

# ---------- LOAD TXT FILE ----------
loader = TextLoader(DATA_PATH, encoding="utf-8")
documents = loader.load()

# ---------- SPLIT TEXT ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# ---------- EMBEDDINGS ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- VECTOR DATABASE ----------
vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory=DB_PATH
)
vectorstore.persist()

# ---------- LLM ----------
llm = Ollama(model=MODEL_NAME)

# ---------- QA CHAIN ----------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ---------- CHAT LOOP ----------
print("📘 Edu Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    result = qa_chain(query)
    print("\n🤖 Bot:", result["result"], "\n")