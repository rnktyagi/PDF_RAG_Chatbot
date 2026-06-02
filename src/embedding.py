from src.ingestion import ingestion_and_chunking

from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct

from fastembed import TextEmbedding, SparseTextEmbedding

import uuid

from dotenv import load_dotenv

import os



load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_url = os.getenv("QDRANT_URL")



def upload_file(file_path: str, user_id: str, collection_name="pdf_rag"):



    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)



    dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")



    # 1. ONLY the database creation should be inside this IF block

    if not client.collection_exists(collection_name):

        client.create_collection(

            collection_name=collection_name,

            vectors_config={

                "dense": VectorParams(size=384, distance=Distance.COSINE)

            },

            sparse_vectors_config={

                "sparse": SparseVectorParams()

            }

        )



    # 2. EVERYTHING ELSE MUST BE UN-INDENTED SO IT RUNS EVERY TIME

    try:

        docs = ingestion_and_chunking(file_path)        

        texts = [doc.page_content for doc in docs]



        dense_vectors = list(dense_model.embed(texts))

        sparse_vectors = list(sparse_model.embed(texts))



        points = []

        file_id = str(uuid.uuid4())



        for i, doc in enumerate(docs):

            # 1. Convert numpy array to standard Python list

            dense_vec = dense_vectors[i].tolist()

           

            # 2. Extract indices and values from FastEmbed's custom object

            sparse_emb = sparse_vectors[i]

            sparse_vec = {

                "indices": sparse_emb.indices.tolist(),

                "values": sparse_emb.values.tolist()

            }

            chunk_id = str(uuid.uuid4())



            point = PointStruct(

                id=chunk_id, # Reusing the same file_id so all chunks tie back to one file

                vector={

                    'dense': dense_vec,

                    'sparse': sparse_vec

                },

                payload={

                    'user_id': user_id,

                    'file_id': file_id,

                    'text': doc.page_content,

                    "source": doc.metadata.get("source"),

                    "pages": doc.metadata.get("pages"),

                    "section": doc.metadata.get("section")

                }

            )

            points.append(point)



        # (Optional but safe) Tell Qdrant to index it just in case

        try:

            client.create_payload_index(

                collection_name=collection_name,

                field_name="user_id",

                field_schema="keyword"

            )

        except Exception:

            pass



        # Send to database

        client.upsert(collection_name=collection_name, points=points)



        # 3. THE LOUD TERMINAL ANNOUNCEMENT

        print("\n" + "="*60, flush=True)

        print(f"✅ SUCCESS: PDF FULLY PROCESSED FOR USER {user_id}", flush=True)

        print("✅ YOU CAN NOW ASK QUESTIONS IN STREAMLIT!", flush=True)

        print("="*60 + "\n", flush=True)



    except Exception as e:

        # 4. IF IT CRASHES, SCREAM THE ERROR TO THE TERMINAL

        print("\n" + "!"*60, flush=True)

        print(f"❌ UPLOAD FAILED SILENTLY IN BACKGROUND:", flush=True)

        print(f"{str(e)}", flush=True)

        print("!"*60 + "\n", flush=True) 

