import os
import openai
import pinecone  # ✅ Use the new Pinecone SDK
from fastapi import FastAPI, HTTPException  # ✅ Add HTTPException
from pydantic import BaseModel

# Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Index Configuration
PINECONE_INDEX_NAME = "anaya-777"
PINECONE_ENVIRONMENT = "us-west-2"
PINECONE_DIMENSION = 3072

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# ✅ Initialize Pinecone (New SDK)
from pinecone import Pinecone, ServerlessSpec  # ✅ Import the new SDK

# ✅ Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Check if index exists before creating
existing_indexes = pc.list_indexes()
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
else:
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists. Skipping creation.")

# ✅ Connect to Pinecone Index
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
        input=lesson.content, model="text-embedding-3-large"
    )['data'][0]['embedding']
    
    index.upsert([(str(lesson.title), embedding, {"content": lesson.content})])  # ✅ Ensure ID is a string
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

# ✅ Retrieve All Lessons (Fixed)
@app.get("/all_lessons/")
def all_lessons():
    lessons = []
    index_stats = index.describe_index_stats()
    for vector_id in index_stats['total_vector_count']:
        vector = index.fetch([vector_id])['vectors'][vector_id]
        lessons.append({"title": vector_id, "content": vector["metadata"]["content"]})
    return {"lessons": lessons}

# ✅ ✅ ✅ FastAPI app now correctly initialized ✅ ✅ ✅
