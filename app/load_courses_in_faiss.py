import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'courses.json')

# Load courses.json (replace with your file path)
# with open('courses.json', 'r') as f:
#     courses = json.load(f)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    courses = json.load(f)

# Create documents for FAISS
documents = [
    Document(
        page_content=f"{course['description']} Tags: {', '.join(course['tags'])}",
        metadata={
            'id': course['id'],
            'title': course['title'],
            'provider': course['provider'],
            'skill_level': course['skill_level'],
            'duration': course['duration'],
            'url': course['url']
        }
    )
    for course in courses
]

# Initialize Sentence Transformers embeddings (downloads model on first run)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Lightweight, English-focused; alternatives: "all-mpnet-base-v2" for better quality (slower)
)

# Create FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)
# Save to disk for reuse (optional)
vectorstore.save_local("faiss_index2")