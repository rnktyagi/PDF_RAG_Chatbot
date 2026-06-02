import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")


class Retriever:
    def __init__(self, collection_name="pdf_rag"):
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        self.dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query: str, user_id: str):
        dense_query_vector = list(self.dense_model.embed([query]))[0]

        sparse_query = list(self.sparse_model.embed([query]))[0]
        sparse_query_vector = models.SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values
        )

        user_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query_vector,
                    limit=20,
                    using="dense",
                    filter=user_filter
                ),
                models.Prefetch(
                    query=sparse_query_vector,
                    using="sparse",
                    limit=20,
                    filter=user_filter
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20
        )

        texts = [
            point.payload.get("text", "")
            for point in results.points
        ]

        rerank_scores = list(self.reranker.rerank(query, texts))

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

        reranked_results.sort(key=lambda x: x["rerank_score"],reverse=True)

        final_top_results = reranked_results[:5]

        return final_top_results