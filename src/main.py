from fastapi import FastAPI , HTTPException , UploadFile, File, BackgroundTasks , Form
from pydantic import BaseModel , Field
import os
from dotenv import load_dotenv
from src.graph import workflow
from src.embedding import upload_file
import shutil
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

load_dotenv()

app = FastAPI(
    title="Enterprise PDF RAG API",
    description="A production-grade backend powering an intelligent LangGraph agent.",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str = Field(..., description="The raw message string from the user.")
    user_id: str = Field(..., description="The unique identifier for the tenant context.")
    thread_id: str = Field(..., description="The unique session ID tracking the short-term chat history.")

@app.post("/chat", summary="Return an answer using the RAG backend to the user query.")
async def chat_endpoint(request: ChatRequest):
    try:
        config = {'configurable': {'thread_id': request.thread_id}}
        initial_state = {
            "messages": [("user", request.message)],
            "user_id": request.user_id
        }
        
        # 1. Grab the database URL
        db_uri = os.getenv("DATABASE_URI")
        
        # 2. Open a fresh, guaranteed-alive connection to Postgres
        with PostgresSaver.from_conn_string(db_uri) as checkpointer:
            
            # (Optional) Ensure tables exist
            checkpointer.setup()
            
            # 3. Compile the LangGraph blueprint with our fresh memory connection
            agent = workflow.compile(checkpointer=checkpointer)
            
            # 4. Run the AI pipeline
            result = agent.invoke(initial_state, config=config)

        # 5. Extract the AI's final answer
        output_messages = result.get("messages", [])
        if not output_messages:
            raise ValueError("No messages returned from the graph.")
            
        ai_response = output_messages[-1].content

        return {
            "status": "success",
            "thread_id": request.thread_id,
            "response": ai_response
        }
        
    except Exception as e:
        print(f"Backend Error: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"Agent Processing Error: {str(e)}")
    
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload", summary="Upload a PDF and process its embeddings in the background")
async def upload_pdf(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    user_id : str = Form(...)
):
    local_file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(upload_file, local_file_path, user_id)

    return {
        "status": "success",
        "message": f"'{file.filename}' received successfully. Ingestion pipeline started in the background."
    }