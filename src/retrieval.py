import os
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings import FastEmbedEmbeddings
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")


class Retriever:
    def __init__(self, collection_name: str = "pdf_rag"):
        self.collection_name = collection_name

        dense_embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # This one object replaces the client, both embed calls, both
        # Prefetch blocks, and the FusionQuery(RRF) call from before.
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query: str, user_id: str, top_k: int = 5):
        user_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user_id",
                    match=models.MatchValue(value=user_id),
                )
            ]
        )

        # Runs the hybrid (dense + sparse, RRF-fused) search under the hood
        results = self.vector_store.similarity_search_with_score(
            query,
            k=20,
            filter=user_filter,
        )

        texts = [doc.page_content for doc, _ in results]
        rerank_scores = list(self.reranker.rerank(query, texts))

        reranked_results = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source"),
                "pages": doc.metadata.get("pages"),
                "section": doc.metadata.get("section"),
                "original_qdrant_score": score,
                "rerank_score": float(rerank_score),
            }
            for (doc, score), rerank_score in zip(results, rerank_scores)
        ]

        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_results[:top_k]