from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch  # Updated import
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os
import operator
import uuid
import json
import sqlite3


load_dotenv()

#LLMs
relevance_checker_llm = ChatCohere(model="command-a-03-2025", cohere_api_key=os.getenv("COHERE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
tavily_tool = TavilySearch(max_results=5, tavily_api_key=os.getenv("TAVILY_API_KEY"))


#Tools
@tool
def query_courses_semantic(query: str, k: int = 5) -> List[Dict]:
    """Perform semantic search on courses using FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "id": doc.metadata['id'],
            "title": doc.metadata['title'],
            "provider": doc.metadata['provider'],
            "skill_level": doc.metadata['skill_level'],
            "duration": doc.metadata['duration'],
            "url": doc.metadata['url'],
            "similarity_score": score
        }
        for doc, score in results
    ]

tools = [tavily_tool, query_courses_semantic]

# State
class AgentState(TypedDict):
    query: str
    is_relevant: bool
    direct_answer_possible: bool
    web_results: Annotated[List[dict], operator.add]
    db_results: Annotated[List[dict], operator.add]
    final_answer: str
    tools_to_call: List[str]
    thread_id: str
    user_id: str
    # disliked_courses: List[str]
    conversation_history: List[Dict[str, str]]

# Node Functions
def relevance_checker(state):
    # Load conversation history for context
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT query, response FROM conversations WHERE thread_id = ? ORDER BY timestamp DESC LIMIT 3", (state['thread_id'],))
    history = [{"query": row[0], "response": row[1]} for row in c.fetchall()]
    conn.close()
    state['conversation_history'] = history
    prompt = ChatPromptTemplate.from_template(
        """Classify if this query is relevant to education, courses, skills, or learning paths, considering the conversation history.
        Query: {query}
        Conversation history (last 3): {history}
        Respond in JSON:
        ```json
        {
            "relevant": "yes" or "no",
            "reason": "Brief explanation"
        }
        ```
        Examples:
        - Query: "Courses for machine learning", History: [] -> {"relevant": "yes", "reason": "Asks about courses"}
        - Query: "Are they on your website?", History: [{"query": "CI/CD learning path", "response": "..."}] -> {"relevant": "yes", "reason": "Refers to previous course-related query"}
        - Query: "What's the weather?", History: [] -> {"relevant": "no", "reason": "Unrelated to education"}"""
    )
    try:
        formatted_prompt = prompt.format(query=state['query'], history=json.dumps(state['conversation_history']))
        print(f"Formatted prompt: {formatted_prompt}")
        response = llm.invoke(formatted_prompt).content
        print(f"Relevance checker response: {response}")
        response_dict = json.loads(response.strip()) if response.startswith('{') else {"relevant": "no", "reason": "Invalid response format"}
        state['is_relevant'] = response_dict.get('relevant', 'no').lower() == 'yes'
        if not state['is_relevant']:
            state['final_answer'] = "Please ask a question related to courses or learning paths."
    except Exception as e:
        print(f"Relevance checker error: {e}")
        state['is_relevant'] = any(keyword in state['query'].lower() for keyword in ['course', 'skill', 'learn', 'education', 'path', 'website'])
        if not state['is_relevant']:
            state['final_answer'] = "Please ask a question related to courses or learning paths."
        else:
            print("Fallback: Query deemed relevant due to keywords")
    return state

def router(state):
    if not state['is_relevant']:
        return state
    prompt = ChatPromptTemplate.from_template(
        """Based on query: {query}
        Conversation history (last 3): {history}
        Decide how to answer: 'direct' (no tools), 'web' (external data), 'db' (course database), or 'both' (web+db).
        Respond in JSON:
        ```json
        {
            "action": "direct" or "web" or "db" or "both",
            "reason": "Brief explanation"
        }
        ```
        Examples:
        - Query: "What is machine learning?" -> {"action": "direct", "reason": "General question"}
        - Query: "Courses for Python" -> {"action": "db", "reason": "Specific course recommendation"}
        - Query: "Are they on your website?", History: [{"query": "CI/CD learning path", ...}] -> {"action": "db", "reason": "Refers to previous courses"}"""
    )
    try:
        formatted_prompt = prompt.format(query=state['query'], history=json.dumps(state['conversation_history']))
        print(f"Router formatted prompt: {formatted_prompt}")
        response = llm.invoke(formatted_prompt).content
        print(f"Router response: {response}")
        response_dict = json.loads(response.strip()) if response.startswith('{') else {"action": "db", "reason": "Default to DB"}
        action = response_dict.get('action', 'db').lower()
        state['direct_answer_possible'] = action == 'direct'
        state['tools_to_call'] = (
            [] if action == 'direct' else
            ['web', 'db'] if action == 'both' else
            ['web'] if action == 'web' else
            ['db']
        )
    except Exception as e:
        print(f"Router error: {e}")
        state['direct_answer_possible'] = False
        state['tools_to_call'] = ['db']
    return state

def web_search(state):
    results_to_add = []
    if 'web' in state.get('tools_to_call', []):
        try:
            results = tavily_tool.invoke(state['query'])
            results_to_add = results if isinstance(results, list) else []
            print(f"Web search returned {len(results_to_add)} results")
        except Exception as e:
            print(f"Web search error: {e}")
    return {"web_results": results_to_add}

def db_query(state):
    results_to_add = []
    if 'db' in state.get('tools_to_call', []):
        try:
            results = query_courses_semantic.invoke({"query": state['query'], "k": 5})
            results_to_add = results
            print(f"DB query returned {len(results_to_add)} results")
        except Exception as e:
            print(f"DB query error: {e}")
    return {"db_results": results_to_add}


def synthesizer(state):
    # Save query and response to SQLite
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (thread_id, user_id, query, response) VALUES (?, ?, ?, ?)",
              (state['thread_id'], state['user_id'], state['query'], state.get('final_answer', '')))
    conn.commit()
    conn.close()

    if not state['db_results'] and not state['web_results'] and not state['direct_answer_possible']:
        state['final_answer'] = "No specific courses or information found. Try a different query."
        return state
    if state['direct_answer_possible']:
        prompt = ChatPromptTemplate.from_template(
            """Answer directly: {query}
            Conversation history (last 3): {history}
            Provide a concise, helpful response in Markdown."""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """Query: {query}
            Conversation history (last 3): {history}
            Web results: {web_results}
            DB courses: {db_results}
            Provide a helpful response in Markdown. Recommend courses, explain skills, or suggest learning paths. Be concise."""
        )
    try:
        state['final_answer'] = llm.invoke(prompt.format(
            query=state['query'],
            history=json.dumps(state['conversation_history']),
            web_results=state['web_results'],
            db_results=state['db_results']
        )).content
    except Exception as e:
        state['final_answer'] = f"Error generating response: {e}. Try again."
    return state

# Build Graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("relevance_checker", relevance_checker)
workflow.add_node("router", router)
workflow.add_node("web_search", web_search)
workflow.add_node("db_query", db_query)
workflow.add_node("synthesizer", synthesizer)

# Edges
workflow.set_entry_point("relevance_checker")
workflow.add_conditional_edges(
    "relevance_checker",
    lambda s: END if not s['is_relevant'] else "router"
)
workflow.add_conditional_edges(
    "router",
    lambda s: "synthesizer" if s['direct_answer_possible'] else "web_search" if s['tools_to_call'] == ['web'] else "db_query" if s['tools_to_call'] == ['db'] else ["web_search", "db_query"]
)
workflow.add_edge("web_search", "synthesizer")
workflow.add_edge("db_query", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# # Interactive testing
# while True:
#     query = input("\nEnter query (or 'quit'): ")
#     if query.lower() == 'quit':
#         break
#     result = app.invoke({"query": query, "web_results": [], "db_results": [], "tools_to_call": []})
#     print("\nResponse:", result['final_answer'])