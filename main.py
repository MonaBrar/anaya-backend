import openai
import pinecone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load API Keys (Replace with your actual keys)
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENVIRONMENT = "your_pinecone_environment"
PINECONE_INDEX_NAME = "anaya-memory"

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Initialize Pinecone Index
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536, metric='cosine')
index = pinecone.Index(PINECONE_INDEX_NAME)

# FastAPI App
app = FastAPI()

# Data Model for Lessons
class Lesson(BaseModel):
    title: str
    content: str

# Store a Lesson in Pinecone
@app.post("/store_lesson/")
def store_lesson(lesson: Lesson):
    embedding = openai.Embedding.create(
        input=lesson.content, model="text-embedding-ada-002"
    )['data'][0]['embedding']
    
    index.upsert([(lesson.title, embedding, {"content": lesson.content})])
    return {"message": f"Lesson '{lesson.title}' stored successfully."}

# Retrieve a Lesson from Pinecone
@app.get("/retrieve_lesson/")
def retrieve_lesson(query: str):
    query_embedding = openai.Embedding.create(
        input=query, model="text-embedding-ada-002"
    )['data'][0]['embedding']
    
    results = index.query(query_embedding, top_k=3, include_metadata=True)
    if not results['matches']:
        raise HTTPException(status_code=404, detail="No relevant lessons found.")
    
    return {"lessons": [{"title": match['id'], "content": match['metadata']['content']} for match in results['matches']]}

# Retrieve All Lessons (For Lesson Tracking)
@app.get("/all_lessons/")
def all_lessons():
    lessons = []
    for vector in index.fetch(ids=index.list_ids())['vectors'].values():
        lessons.append({"title": vector['id'], "content": vector['metadata']['content']})
    return {"lessons": lessons}
