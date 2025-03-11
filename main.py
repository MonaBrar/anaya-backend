import os
import openai
import pinecone
from pinecone import ServerlessSpec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Root Route
@app.get("/")
def read_root():
    return {"message": "FastAPI is live and connected to Pinecone!"}

# ✅ Define the Lesson Model
class Lesson(BaseModel):
    title: str
    content: str

# ✅ Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-west-2"  # Updated for the new index
PINECONE_INDEX_NAME = "anaya444s"  # ✅ Use the new index name

# ✅ Ensure API keys are set
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys! Please set environment variables.")

# ✅ Initialize OpenAI
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize Pinecone and connect to the new index
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# ✅ Ensure the new index exists
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # ✅ Correct dimension for text-embedding-ada-002
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pinecone.Index(PINECONE_INDEX_NAME)

# ✅ Store a Lesson in Pinecone
@app.post("/store_lesson/")
async def store_lesson(lesson: Lesson):
    response = openai_client.embeddings.create(
        input=lesson.content,
        model="text-embedding-ada-002"
    )
    embedding_vector = response.data[0].embedding

    # ✅ Store in Pinecone
    index.upsert([(lesson.title, embedding_vector, {"content": lesson.content})])

    return {"message": f"Lesson '{lesson.title}' stored successfully."}

# ✅ Retrieve a Lesson
@app.get("/retrieve_lesson/")
async def retrieve_lesson(query: str):
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding

    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    if not results['matches']:
        raise HTTPException(status_code=404, detail="No relevant lessons found.")

    lessons = [{"title": match["id"], "content": match['metadata']['content']} for match in results['matches']]

    return {"lessons": lessons}

# ✅ Retrieve All Lessons
@app.get("/all_lessons/")
def get_all_lessons():
    vectors = index.query(
        vector=[0.0] * 1536,
        top_k=100,
        include_metadata=True
    )['matches']

    lessons = [{"title": vector["id"], "content": vector["metadata"]["content"]} for vector in vectors]

    return {"lessons": lessons}
