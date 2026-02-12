import streamlit as st

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_PATH = "chroma_db"
MODEL_NAME = "llama3.2"

# Page config
st.set_page_config(
    page_title="Edu Chatbot",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Educational Chatbot")
st.caption("Powered by Ollama + LangChain + ChromaDB")

# Load embeddings and vector DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

llm = Ollama(model=MODEL_NAME)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask a question from the syllabus...")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain(prompt)["result"]
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )