"""
build_knowledge_base.py
Script to build a RAG knowledge base from course data, with optional web augmentation using Tavily API.
"""
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.database import get_all_courses

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.faiss")

def fetch_web_knowledge_for_course(course, tavily_api_key=None):
    """Fetches web snippets for a course using Tavily API."""
    if not TAVILY_AVAILABLE or not tavily_api_key:
        return None
    client = TavilyClient(api_key=tavily_api_key)
    query = f"Learning path for {course.title} {course.description}"
    try:
        web_results = client.search(query=query, max_results=2)
        return "\n---\n".join([r["content"] for r in web_results["results"]])
    except Exception as e:
        return f"[Tavily error: {e}]"

def build_knowledge_base(gemini_api_key=None, tavily_api_key=None):
    courses = get_all_courses()
    docs = []
    for c in courses:
        base_content = f"{c.title}\n{c.description}\nTags: {', '.join(c.tags)}\nProvider: {c.provider}"
        web_knowledge = fetch_web_knowledge_for_course(c, tavily_api_key)
        if web_knowledge:
            content = f"{base_content}\nWeb Knowledge:\n{web_knowledge}"
        else:
            content = base_content
        docs.append(Document(page_content=content, metadata={"id": c.id, "title": c.title, "provider": c.provider}))

    # Add curated Q&A or learning path docs if desired
    # Example:
    # docs.append(Document(page_content="...", metadata={...}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    split_docs = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=gemini_api_key or os.getenv("GOOGLE_API_KEY"))
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Knowledge base built and saved to {VECTORSTORE_PATH}")

def ensure_knowledge_base(gemini_api_key=None, tavily_api_key=None):
    """Builds the knowledge base if it does not exist."""
    if not os.path.exists(VECTORSTORE_PATH):
        build_knowledge_base(gemini_api_key, tavily_api_key)
    else:
        print(f"Knowledge base already exists at {VECTORSTORE_PATH}")

if __name__ == "__main__":
    build_knowledge_base()
