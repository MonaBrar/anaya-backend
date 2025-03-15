from pymongo import MongoClient
import os

# Load MongoDB Connection String from Environment Variables
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MongoDB connection string is missing! Please set MONGO_URI.")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client.anaya_memory  # Create a new database for Anaya’s memory
conversations_collection = db.conversations  # Collection for storing conversations

import streamlit as st

st.title("Anaya: Your Personal Guide")
st.write("Welcome to Anaya's learning system. Let's begin!")

if st.button("Click Me"):
    st.write("Hello, Monika! Anaya is listening. 💛✨")
