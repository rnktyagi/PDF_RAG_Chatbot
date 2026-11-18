
---
title: PDF RAG Backend
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# PDF RAG Chatbot

A PDF-based question-answering chatbot that allows users to upload PDF documents and ask questions in natural language. It uses Retrieval-Augmented Generation (RAG) to provide relevant answers based on the uploaded documents.

## Features

- **PDF Upload:** Upload PDF documents and process their content.
- **Question Answering:** Ask questions about the uploaded documents.
- **Context-Based Answers:** Generates answers using information retrieved from the PDFs.
- **Page-Level Citations:** Provides references to relevant pages in the documents.
- **Conversation Memory:** Maintains context across multiple questions.
- **Suggested Questions:** Recommends follow-up questions based on the conversation.
- **Smart Retrieval:** Rewrites questions and retrieves relevant information to improve answer quality.
- **Data Privacy:** Supports redaction of sensitive information such as email addresses and card numbers.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| FastAPI | Backend API |
| LangChain | LLM application framework |
| LangGraph | Workflow and conversation management |
| Docling | PDF document processing |
| Qdrant | Vector database |
| PostgreSQL | Conversation checkpoint storage |
| Groq | LLM integration |
| Docker | Application deployment |

## How It Works

1. **Upload PDF:** The user uploads a PDF document.
2. **Document Processing:** The PDF is parsed and divided into smaller chunks.
3. **Vector Storage:** Document chunks are converted into embeddings and stored in Qdrant.
4. **Question Processing:** The user's question is cleaned and rewritten when required.
5. **Information Retrieval:** Relevant document chunks are retrieved from the vector database.
6. **Answer Generation:** The LLM generates an answer using the retrieved information.
7. **Response:** The chatbot displays the answer with relevant citations and suggested follow-up questions.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rnktyagi/PDF_RAG_Chatbot.git
cd PDF_RAG_Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

> **Note:** The application also communicates with a separately deployed FastAPI backend. The backend URL and required environment variables must be configured according to your deployment.

## API Endpoints

The frontend communicates with the backend through the following endpoints:

### Upload PDF

**Endpoint:** `/upload`

**Method:** `POST`

Accepts a PDF file and a user ID for document processing.

### Chat

**Endpoint:** `/chat`

**Method:** `POST`

Accepts a user message, user ID, and conversation thread ID.

Example request:

```json
{
  "message": "What is this document about?",
  "user_id": "user123",
  "thread_id": "thread123"
}
```

## Project Use Cases

- Understanding lengthy PDF documents.
- Finding specific information in reports and notes.
- Asking questions about academic study material.
- Extracting relevant information from documentation.
- Interacting with documents through natural language.

## Future Improvements

- Support for additional document formats.
- Improved document retrieval and answer accuracy.
- Enhanced user authentication.
- Better conversation management.

## Author

**Raunak Tyagi**

GitHub: (https://github.com/rnktyagi)

## Contributors
**Rajbir Singh**
Github: (https://github.com/rjvrr)

**Ranti Dev**
Github: (https://github.com/rantidev1802)

## Repository

[PDF RAG Chatbot](https://github.com/rnktyagi/PDF_RAG_Chatbot)
