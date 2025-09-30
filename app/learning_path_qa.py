
"""
learning_path_qa.py
Q&A agent for learning path questions using LangChain, Gemini, and course-based retrieval.
"""
import os
from typing import Optional
from app.database import get_all_courses

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings

# Optional: Tavily for web augmentation
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert career and learning path advisor. "
    "Answer user questions about learning paths, programming languages, and career steps. "
    "Be concise, factual, and context-aware."
)


class LearningPathQnA:
    def __init__(self, gemini_api_key: Optional[str] = None, tavily_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.gemini_api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.gemini_api_key)
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        if TAVILY_AVAILABLE and self.tavily_api_key:
            self.tavily = TavilyClient(api_key=self.tavily_api_key)
        else:
            self.tavily = None

    def _load_vectorstore(self):
        VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.faiss")
        if os.path.exists(VECTORSTORE_PATH):
            return FAISS.load_local(VECTORSTORE_PATH, self.embeddings)
        else:
            raise RuntimeError(f"Knowledge base not found at {VECTORSTORE_PATH}. Please build it first.")

    def answer(self, question: str, user_context: Optional[str] = None, use_web: bool = False) -> str:
        """Answer a learning path question using Gemini and course retrieval. Optionally augment with Tavily web search."""
        context_docs = self.retriever.invoke(question)
        context = "\n---\n".join([doc.page_content for doc in context_docs])
        if user_context:
            context = f"User context: {user_context}\n{context}"

        web_snippets = ""
        if use_web and self.tavily:
            try:
                web_results = self.tavily.search(query=question, max_results=3)
                web_snippets = "\n---\n".join([r["content"] for r in web_results["results"]])
            except Exception as e:
                web_snippets = f"[Tavily error: {e}]"

        prompt = ChatPromptTemplate.from_messages([
            ("system", DEFAULT_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\nWeb:\n{web_snippets}\nQuestion: {question}\nAnswer:")
        ])
        chain = (
            RunnableParallel({
                "context": lambda x: context,
                "web_snippets": lambda x: web_snippets,
                "question": lambda x: question
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        try:
            return chain.invoke({"context": context, "web_snippets": web_snippets, "question": question})
        except Exception as e:
            return f"Error: {e}"
