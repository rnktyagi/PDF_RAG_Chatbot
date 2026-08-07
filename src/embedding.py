import uuid
import os
from dotenv import load_dotenv
from src.ingestion import ingestion_and_chunking
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

def upload_file(file_bytes: bytes, filename: str, user_id: str, collection_name: str = "pdf_rag"):
    docs = ingestion_and_chunking(file_bytes, filename)

    file_id = str(uuid.uuid4())
    for doc in docs:
        doc.metadata["user_id"] = user_id
        doc.metadata["file_id"] = file_id

    vector_store = QdrantVectorStore.from_documents(
        docs,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    try:
        vector_store.client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.user_id",
            field_schema="keyword",
        )
    except Exception:
        print("Failed")
    return vector_store