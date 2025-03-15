import os
import openai
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from neo4j import GraphDatabase, basic_auth

# FastAPI app initialization
app = FastAPI()

# ✅ Neo4j Credentials
NEO4J_URI = "neo4j+s://eda4629b.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "hcNZkmmtT6xIdGw-PZ5yRyXCTCS6dX6ezNjFexfEr4k"

# ✅ Create Neo4j Driver (Only Once!)
driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))

# ✅ Function to Test Neo4j Connection
def test_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j connection successful' AS message")
            for record in result:
                print(record["message"])
    except Exception as e:
        print(f"🚨 Neo4j Connection Error: {e}")

# ✅ Run Connection Test Once at Startup
test_connection()

# ✅ Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "anaya444s"

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Missing API keys! Please set environment variables.")

# ✅ OpenAI Initialization
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure Pinecone Index Exists
existing_indexes = [index.name for index in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Data Model for Lessons
class Lesson(BaseModel):
    title: str
    content: str

# ✅ Store a Lesson in Pinecone & Neo4j
@app.post("/store_lesson/")
async def store_lesson(lesson: Lesson):
    try:
        # ✅ Generate Embedding
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=lesson.content
        )
        embedding = response.data[0].embedding

        # ✅ Store in Pinecone
        index.upsert([
            (lesson.title, embedding, {"content": lesson.content})
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI/Pinecone Error: {str(e)}")

    # ✅ Store in Neo4j
    query = """
    MERGE (l:Lesson {title: $title})
    SET l.content = $content
    RETURN l
    """
    try:
        with driver.session() as session:
            session.run(query, title=lesson.title, content=lesson.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j Error: {str(e)}")

    return {"message": f"Lesson '{lesson.title}' stored successfully in Pinecone and Neo4j."}

# ✅ Retrieve a Lesson from Pinecone
@app.get("/retrieve_lesson/")
async def retrieve_lesson(query: str):
    try:
        print(f"🔍 Incoming Query: {query}")

        # ✅ Trim whitespace and remove newline characters
        cleaned_query = query.strip()
        if not cleaned_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        response = openai_client.embeddings.create(
            input=cleaned_query,
            model="text-embedding-ada-002"
        )

        query_embedding = np.array(response.data[0].embedding, dtype=np.float32).tolist()

        print(f"✅ Query Embedding Generated Successfully")

        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        print(f"📡 Pinecone Query Results: {results}")

        if not results.matches:
            raise HTTPException(status_code=404, detail="No relevant lessons found.")

        lessons = [{"title": match["id"], "content": match["metadata"]["content"]} for match in results.matches]

        return {"lessons": lessons}

    except Exception as e:
        print(f"🚨 Retrieval Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pinecone Retrieval Error: {str(e)}")

# ✅ Retrieve All Lessons from Pinecone (Fixed Query Syntax)
@app.get("/all_lessons/")
async def all_lessons():
    dummy_vector = np.random.rand(1536).tolist()
    results = index.query(
        vector=dummy_vector,  # ✅ FIXED Query Syntax
        top_k=100,
        include_metadata=True
    )

    lessons = [
        {"title": match["id"], "content": match["metadata"]["content"]}
        for match in results["matches"]
    ]

    return {"lessons": lessons}

# ✅ Test Endpoint
@app.get("/test")
def test_api():
    return {"message": "API is running smoothly!"}

# ✅ Store Lesson in Neo4j Only
@app.post("/store_lesson_neo4j/")
async def store_lesson_neo4j(lesson: Lesson):
    query = """
    MERGE (l:Lesson {title: $title})
    SET l.content = $content
    RETURN l
    """
    try:
        with driver.session() as session:
            session.run(query, title=lesson.title, content=lesson.content)
        return {"message": f"Lesson '{lesson.title}' stored in Neo4j successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j Error: {str(e)}")

# ✅ Debug Route to List Available Endpoints
@app.get("/routes")
async def get_routes():
    return {"routes": [route.path for route in app.routes]}
