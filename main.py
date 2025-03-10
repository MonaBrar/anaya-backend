import os
import openai
from pinecone import Pinecone, ServerlessSpec  # ✅ Correct Pinecone import
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # ✅ Keep this import!

# ✅ FastAPI app initialization
app = FastAPI()

# ✅ Root Route to confirm API is live
@app.get("/")
def read_root():
    return {"message": "Welcome to Anaya's FastAPI backend!"}

# ✅ Data model for lesson storage
class Lesson(BaseModel):
    title: str
    content: str

# ✅ Set your API keys securely (ensure they are set in Render)
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys! Please set environment variables.")

# ✅ OpenAI Initialization (new syntax)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize Pinecone
from pinecone import ServerlessSpec

PINECONE_INDEX_NAME = "anaya-memory"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone Index
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Route to store a lesson in Pinecone
@app.post("/store_lesson/")
async def store_lesson(lesson: Lesson):
    response = openai_client.embeddings.create(
        input=lesson.content,
        model="text-embedding-ada-002"
    )
    embedding_vector = response.data[0].embedding

    # ✅ Extend embedding if required by Pinecone dimension
    extended_embedding = embedding_vector + embedding_vector

    # ✅ Upsert into Pinecone
    index.upsert([(lesson.title, extended_embedding, {"content": lesson.content})])

    return {"message": f"Lesson '{lesson.title}' stored successfully."}

# ✅ Route to retrieve lessons
@app.get("/retrieve_lesson/")
async def retrieve_lesson(query: str):
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding

    # ✅ Query Pinecone
    results = index.query(queries=[query_embedding], top_k=3, include_metadata=True)

    if not results['matches']:
        raise HTTPException(status_code=404, detail="No relevant lessons found.")

    lessons = [{"title": match["id"], "content": match['metadata']['content']} for match in results['matches']]

    return {"lessons": lessons}

# ✅ Route to retrieve all lessons (for verification purposes)
@app.get("/all_lessons/")
def get_all_lessons():
    vectors = index.query(
        vector=[0.0]*3072,
        top_k=100,
        include_metadata=True
    )['matches']

    lessons = [{"title": vector["id"], "content": vector["metadata"]["content"]} for vector in vectors]

    return {"lessons": lessons}
