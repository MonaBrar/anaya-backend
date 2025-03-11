import os
import numpy as np
from openai import OpenAI  # ✅ Correct OpenAI import
from pinecone import Pinecone, ServerlessSpec  # ✅ Correct Pinecone import
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Root Route to confirm API is live
@app.get("/")
def read_root():
    return {"message": "Welcome to Anaya's FastAPI backend!"}

# ✅ Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "anaya444s"

# ✅ Ensure API keys exist
if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Missing API keys! Please set them in the environment variables.")

# ✅ Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)  

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure Pinecone Index Exists
existing_indexes = [index.name for index in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # ✅ Make sure this matches your OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Define Data Model for Lessons
class Lesson(BaseModel):
    title: str
    content: str

# ✅ Store a Lesson in Pinecone
@app.post("/store_lesson/")
async def store_lesson(lesson: Lesson):
    response = client.embeddings.create(
        input=lesson.content,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding

    # ✅ Store in Pinecone with metadata
    index.upsert([(lesson.title, embedding, {"content": lesson.content})])

    return {"message": f"Lesson '{lesson.title}' stored successfully."}

# ✅ Retrieve a Lesson from Pinecone
@app.get("/retrieve_lesson/")
def retrieve_lesson(query: str):
    response = client.embeddings.create(
        input=query, 
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding

    results = index.query(
        queries=[query_embedding],  
        top_k=3,  
        include_metadata=True
    )

    if not results['matches']:
        raise HTTPException(status_code=404, detail="No relevant lessons found.")

    return {"lessons": [
        {"title": match['id'], "content": match['metadata']['content']}
        for match in results['matches']
    ]}

# ✅ Retrieve All Lessons
@app.get("/all_lessons/")
def all_lessons():
    # ✅ Generate a valid random query vector instead of using all zeroes
    dummy_vector = np.random.rand(1536).tolist()

    results = index.query(
        queries=[dummy_vector],  
        top_k=100,  
        include_metadata=True
    )['matches']

    lessons = [
        {"title": match["id"], "content": match["metadata"]["content"]}
        for match in results
    ]

    return {"lessons": lessons}

# ✅ Test Endpoint
@app.get("/test")
def test_api():
    return {"message": "API is running smoothly!"}
