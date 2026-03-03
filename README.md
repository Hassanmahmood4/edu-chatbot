📚 Edu-Chatbot — AI-Powered Educational Chat Assistant

An AI-driven educational chatbot that answers student queries contextually using local LLM inference and semantic knowledge retrieval. Built with LangChain for workflow orchestration, Ollama for fast offline language model responses, and ChromaDB for vector search of knowledge content.  ￼



🚀 Features
	•	🤖 AI Q&A Assistant — Responds to academic questions in natural language
	•	📌 Semantic Search — Uses vector embeddings for context-aware retrieval
	•	🧠 Local LLM Inference — Runs LLMs locally via Ollama (no cloud APIs)
	•	📄 Knowledge Base — Store and search domain knowledge for better answers
	•	🔐 Privacy-First — Designed to keep all processing on your machine
	•	📊 Extensible — Can be adapted for tutoring, documentation search, or FAQ bots  ￼



🛠️ Tech Stack
	•	Python — Core language
	•	LangChain — Orchestrates AI prompt flows
	•	Ollama — Local LLM runtime (e.g., llama3)
	•	ChromaDB — Vector database for embeddings
	•	Hugging Face — Embedding models and optional downstream models  ￼



📦 Quick Start

git clone https://github.com/Hassanmahmood4/edu-chatbot.git
cd edu-chatbot

# Install dependencies
pip install -r requirements.txt

# Pull a local LLM
ollama pull llama3  # or any preferred supported model

# Run the app
python app.py

Open the app in your browser at: http://localhost:PORT (as shown in your terminal).  ￼



🧠 Use Cases
	•	🎓 Student academic support
	•	📘 Document-based tutoring
	•	📝 Study helper / FAQ bot
	•	📚 Knowledge retrieval applications  ￼



📁 Project Structure

edu-chatbot/
├── app.py            # Main app logic
├── chatbot.py        # AI query handling
├── ingest.py         # Knowledge ingestion module
├── knowledge.txt     # Reference data for responses
├── requirements.txt  # Dependencies
└── chroma_db/        # Vector store (auto-generated)



⚡ Recommendations & Next Steps

✔ Add document upload for tutoring corpora
✔ Build a Streamlit/Gradio UI for interactive use
✔ Add user authentication and session history
✔ Support multiple knowledge sources (PDF, TXT, web)  ￼

