import os
import openai
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np  # ✅ Ensure this is imported at the top

# FastAPI app initialization
app = FastAPI()

# Root Route to confirm API is live
@app.get("/")
def read_root():
    return {"message": "Welcome to Anaya's FastAPI backend!"}

# Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "anaya444s"

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Missing API keys! Please set environment variables.")

# OpenAI Initialization
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists
existing_indexes = [index.name for index in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Correct dimension for text-embedding-ada-002
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Data Model for Lessons
class Lesson(BaseModel):
    title: str
    content: str

# Store a Lesson in Pinecone
@app.post("/store_lesson/")
async def store_lesson(lesson: Lesson):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=lesson.content
    )
    embedding = response.data[0].embedding

    index.upsert([
        (lesson.title, embedding, {"content": lesson.content})
    ])

    return {"message": f"Lesson '{lesson.title}' stored successfully."}

# Retrieve a Lesson from Pinecone
@app.get("/retrieve_lesson/")
async def retrieve_lesson(query: str):
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    
    query_embedding = list(map(float, response.data[0].embedding))  # ✅ Ensure it's a list of floats

    results = index.query(
        vector=query_embedding,  # ✅ Fix here!
        top_k=3,
        include_metadata=True
    )

    if not results.matches:
        raise HTTPException(status_code=404, detail="No relevant lessons found.")

    lessons = [{"title": match["id"], "content": match["metadata"]["content"]} for match in results.matches]

    return {"lessons": lessons}

# Retrieve All Lessons from Pinecone
@app.get("/all_lessons/")
async def all_lessons():
    # ✅ Generate a random vector (Pinecone rejects zero vectors)
    dummy_vector = np.random.rand(1536).tolist()

    results = index.query(
        queries=[dummy_vector],  # ✅ Fix here!
        top_k=100,
        include_metadata=True
    )

    lessons = [
        {"title": match["id"], "content": match["metadata"]["content"]}  # ✅ Fix here!
        for match in results["matches"]
    ]

    return {"lessons": lessons}

# Test Endpoint
@app.get("/test")
def test_api():
    return {"message": "API is running smoothly!"}
