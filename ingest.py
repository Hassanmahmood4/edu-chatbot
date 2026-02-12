import streamlit as st

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/knowledge.txt"
DB_PATH = "chroma_db"

# Load text file
loader = TextLoader(DATA_PATH, encoding="utf-8")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory=DB_PATH
)

vectorstore.persist()
print("✅ Knowledge ingested successfully!")