import os
import openai
from pinecone import Pinecone, ServerlessSpec  # ✅ Ensure correct Pinecone import
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ Initialize FastAPI app
app = FastAPI()

# Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Root Route to check if API is live
@app.get("/")
def read_root():
    return {"message": "FastAPI is live on Render!"}

# ✅ Define your other API endpoints below
class Lesson(BaseModel):
    title: str
    content: str

@app.post("/store_lesson/")
def store_lesson(lesson: Lesson):
    return {"message": f"Lesson '{lesson.title}' stored successfully."}

# Pinecone Index Configuration
PINECONE_INDEX_NAME = "anaya-777"  # ✅ Matches your Pinecone index
PINECONE_ENVIRONMENT = "us-west-2"  # ✅ Matches your Pinecone region
PINECONE_DIMENSION = 3072  # ✅ Matches your Pinecone index settings

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# ✅ Initialize Pinecone Client (Fix)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure Index Exists & Connect
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(  # ✅ Correct Pinecone Spec Usage
            cloud="aws", 
            region=PINECONE_ENVIRONMENT
        )
    )

# Connect to Pinecone Index
index = pc.Index(PINECONE_INDEX_NAME)

# ✅ FastAPI App
app = FastAPI()

# ✅ Data Model for Lessons
class Lesson(BaseModel):
    title: str
    content: str

# ✅ Store a Lesson in Pinecone
@app.post("/store_lesson/")
def store_lesson(lesson: Lesson):
embedding = openai.Embedding.create(
    input=lesson.content,
    model="text-embedding-ada-002"
)['data'][0]['embedding']

index.upsert([(lesson.title, embedding, {"content": lesson.content})])

return {"message": f"Lesson '{lesson.title}' stored successfully."}

# ✅ Retrieve a Lesson from Pinecone
@app.get("/retrieve_lesson/")
def retrieve_lesson(query: str):
    query_embedding = openai.Embedding.create(
        input=query, model="text-embedding-3-large"  # ✅ Consistent embedding model
    )['data'][0]['embedding']
    
    results = index.query(query_embedding, top_k=3, include_metadata=True)
    if not results['matches']:
        raise HTTPException(status_code=404, detail="No relevant lessons found.")
    
    return {"lessons": [{"title": match['id'], "content": match['metadata']['content']} for match in results['matches']]}

# ✅ Retrieve All Lessons (For Lesson Tracking)
@app.get("/all_lessons/")
def all_lessons():
    lessons = []
    for vector in index.query(queries=[[0] * PINECONE_DIMENSION], top_k=100, include_metadata=True)['matches']:
        lessons.append({"title": vector["id"], "content": vector["metadata"]["content"]})
    return {"lessons": lessons}

# ✅ ✅ ✅ FastAPI app now correctly initialized ✅ ✅ ✅
