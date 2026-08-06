import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.graph import build_agent, AgentContext
from src.embedding import upload_file
import asyncio

load_dotenv()

DB_URI = os.getenv("DATABASE_URI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup() 
        app.state.agent = build_agent(checkpointer)
        yield
    
app = FastAPI(
    title="Enterprise PDF RAG API",
    description="A production-grade backend powering an agentic LangChain RAG assistant.",
    version="2.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The raw message string from the user.")
    user_id: str = Field(..., min_length=1, description="The unique identifier for the tenant context.")
    thread_id: str = Field(..., min_length=1, description="The unique session ID tracking the short-term chat history.")


class ChatResponse(BaseModel):
    status: str
    thread_id: str
    response: str


@app.post("/chat",response_model=ChatResponse,summary="Return an answer using the RAG backend to the user query.",)
async def chat_endpoint(request: ChatRequest):
    agent = app.state.agent  

    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config=config,
            context=AgentContext(user_id=request.user_id),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Agent processing error") from e

    output_messages = result.get("messages", [])
    if not output_messages:
        raise HTTPException(status_code=500, detail="No messages returned from the agent.")

    return ChatResponse(
        status="success",
        thread_id=request.thread_id,
        response=output_messages[-1].content,
    )


@app.post("/upload", summary="Upload a PDF and process its embeddings in the background",)
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(..., min_length=1),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")


    await asyncio.to_thread(
        upload_file,
        file_bytes,
        file.filename,
        user_id,
    )

    return {
        "status": "success",
        "message": "Document processed. You can now ask questions."
    }