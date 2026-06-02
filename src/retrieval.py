import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from fastembed import SparseTextEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
hf_token = os.getenv("HF_TOKEN")

class Retriever() :
    def __init__(self , collection_name = 'pdf_rag_v3') :
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url , api_key=qdrant_api_key)

        # 🚨 THE FIX: Do NOT load models here. Let the server boot fast and light.
        self.dense_model = None
        self.sparse_model = None

    def cloud_rerank(self, query, texts):
        API_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": texts
            }
        }
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Cloud reranker failed: {e}")
            pass
        
        return [0.0] * len(texts)


    def retrieve(self , query : str , user_id : str) :
        # 🚨 THE FIX: Lazy Load. Only turn the models on the very first time someone asks a question!
        if self.dense_model is None:
            self.dense_model = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        if self.sparse_model is None:
            self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        dense_query_vector = self.dense_model.embed_query(query)

        sparse_query = list(self.sparse_model.embed([query]))[0]
        sparse_query_vector = models.SparseVector(indices=sparse_query.indices,
                                                  values=sparse_query.values)
        
        user_filter = models.Filter(must=[models.FieldCondition(key="user_id" , match=models.MatchValue(value=user_id))])

        results = self.client.query_points(collection_name=self.collection_name,
                                           prefetch=[models.Prefetch(
                                               query=dense_query_vector,
                                               limit=20,
                                               using='dense',
                                               filter=user_filter
                                           ),
                                           models.Prefetch(
                                               query=sparse_query_vector,
                                               using='sparse',
                                               limit=20,
                                               filter=user_filter
                                           )],
                                           query=models.FusionQuery(fusion=models.Fusion.RRF),
                                           limit=20)
        
        texts = [point.payload.get('text' , '') for point in results.points]

        rerank_scores = self.cloud_rerank(query, texts)

        reranked_results = []
        for point, score in zip(results.points, rerank_scores):
            reranked_results.append({
                "text": point.payload.get("text"),
                "source": point.payload.get("source"),
                "pages": point.payload.get("pages"),
                "section": point.payload.get("section"),
                "original_qdrant_score": point.score,
                "rerank_score": float(score)
            })

        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_results[:5]