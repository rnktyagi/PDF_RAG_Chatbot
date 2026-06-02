import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# LOOK AT YOUR retrieval.py FILE AND COPY THE EXACT COLLECTION NAME HERE
COLLECTION_NAME = "pdf_rag" 

print(f"Attempting to build index for '{COLLECTION_NAME}'...")

try:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema="keyword"
    )
    print("✅ Index built successfully! Qdrant is ready.")
except Exception as e:
    print(f"❌ FAILED: {e}")