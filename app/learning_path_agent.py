"""
learning_path_agent.py
LangChain agent for learning path Q&A with Gemini, course data, and optional Tavily web search.
"""
import os
from typing import Optional
from app.database import get_all_courses
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.faiss")

class LearningPathAgent:
    def __init__(self, gemini_api_key: Optional[str] = None, tavily_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=self.gemini_api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.gemini_api_key)
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        if TAVILY_AVAILABLE and self.tavily_api_key:
            self.tavily = TavilyClient(api_key=self.tavily_api_key)
        else:
            self.tavily = None

    def _load_vectorstore(self):
        if os.path.exists(VECTORSTORE_PATH):
            return FAISS.load_local(VECTORSTORE_PATH, self.embeddings)
        else:
            raise RuntimeError(f"Knowledge base not found at {VECTORSTORE_PATH}. Please build it first.")

    def _is_simple_query(self, question: str) -> bool:
        # Heuristic: if question is short and not about learning paths/careers, treat as simple
        simple_keywords = ["what is", "define", "explain", "describe", "list"]
        if any(kw in question.lower() for kw in simple_keywords) and len(question.split()) < 12:
            return True
        return False

    def _should_use_tavily(self, question: str) -> bool:
        # Heuristic: if question is about latest trends, best, compare, or not answerable from courses
        web_keywords = ["latest", "best", "compare", "current", "2025", "up to date", "recent"]
        return any(kw in question.lower() for kw in web_keywords)

    def _get_tavily_snippets(self, question: str) -> str:
        if not self.tavily:
            return ""
        try:
            web_results = self.tavily.search(query=question, max_results=3)
            return "\n---\n".join([r["content"] for r in web_results["results"]])
        except Exception as e:
            return f"[Tavily error: {e}]"

    def answer(self, question: str, user_context: Optional[str] = None) -> str:
        # Step 1: LLM node - decide if simple
        if self._is_simple_query(question):
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer concisely."),
                ("human", "{question}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"question": question})

        # Step 2: Decide if Tavily is needed
        use_tavily = self._should_use_tavily(question)
        web_snippets = self._get_tavily_snippets(question) if use_tavily else ""

        # Step 3: Retrieve from course knowledge base
        context_docs = self.retriever.invoke(question)
        context = "\n---\n".join([doc.page_content for doc in context_docs])
        if user_context:
            context = f"User context: {user_context}\n{context}"

        # Step 4: Compose final prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert career and learning path advisor. Use the provided context and web results if available. Be concise, factual, and context-aware."),
            ("human", "Context:\n{context}\nWeb:\n{web_snippets}\nQuestion: {question}\nAnswer:")
        ])
        chain = (
            prompt | self.llm | StrOutputParser()
        )
        return chain.invoke({"context": context, "web_snippets": web_snippets, "question": question})
