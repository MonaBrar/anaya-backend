import os
import openai
from pinecone import Pinecone, ServerlessSpec  # ✅ Correct Pinecone import
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # ✅ Keep this import!

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Root Route to check if API is live
@app.get("/")
def read_root():
    return {"message": "FastAPI is live on Render!"}

# ✅ Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "anaya-memory"

# ✅ Ensure API keys are set
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys! Please set environment variables.")

# ✅ Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# ✅ Initialize Pinecone Properly
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure Index Exists & Connect
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,  # ✅ Matches your Pinecone index settings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Data Model for Lessons
class Lesson(BaseModel):
    title: str
    content: str

# ✅ Store a Lesson in Pinecone
@app.post("/store_lesson/")
async def store_lesson(lesson: Lesson):
    client = openai.OpenAI()
    embedding = client.embeddings.create(
        input=lesson.content,
        model="text-embedding-ada-002"
    ).data[0].embedding

    # ✅ Ensure embedding matches 3072 dimensions
    extended_embedding = embedding + embedding

    # Store in Pinecone
    index.upsert([(lesson.title, extended_embedding, {"content": lesson.content})])

    return {"message": f"Lesson '{lesson.title}' stored successfully."}

# ✅ Retrieve a Lesson from Pinecone
@app.get("/retrieve_lesson/")
def retrieve_lesson(query: str):
    query_embedding = openai.Embedding.create(
        input=query, model="text-embedding-3-large"
    )['data'][0]['embedding']
    
    results = index.query(query_embedding, top_k=3, include_metadata=True)
    if not results['matches']:
        raise HTTPException(status_code=404, detail="No relevant lessons found.")
    
    return {"lessons": [{"title": match['id'], "content": match['metadata']['content']} for match in results['matches']]}

# ✅ Retrieve All Lessons
@app.get("/all_lessons/")
def all_lessons():
    lessons = []
    for vector in index.query(queries=[[0] * 3072], top_k=100, include_metadata=True)['matches']:
        lessons.append({"title": vector["id"], "content": vector["metadata"]["content"]})
    return {"lessons": lessons}

# ✅ ✅ ✅ FastAPI app now correctly initialized ✅ ✅ ✅

