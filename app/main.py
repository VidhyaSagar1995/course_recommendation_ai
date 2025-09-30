import uuid
from typing import Union
from fastapi import FastAPI, HTTPException
from app.recommender import recommend_courses
from app.database import save_feedback, get_user_feedback, init_feedback_db

# Ensure knowledge base is built at startup
from app.build_knowledge_base import ensure_knowledge_base
# from app.learning_path_qa import LearningPathQnA

# # Build or load knowledge base at startup
# import os


# # Q&A agent instance (will use Gemini and the built knowledge base)
# qa_agent = LearningPathQnA()

from app.models import StudentProfile, RecommendationResponse, ParagraphProfile, Feedback, QueryRequest
from app.qa_bot import app as qa_bot_app
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:8080"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(student: Union[StudentProfile, ParagraphProfile]):
    """Recommend courses for a student profile."""
    return recommend_courses(student)


@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    save_feedback(feedback.user_id, feedback.course_id, feedback.feedback)
    return {"status": "success", "message": f"Feedback saved for {feedback.course_id}"}



from app.database import init_db
# Initialize feedback and courses DB on startup
@app.on_event("startup")
def startup_event():
    init_db()
    # ensure_knowledge_base(
    #     gemini_api_key=os.getenv("GOOGLE_API_KEY"),
    #     tavily_api_key=os.getenv("TAVILY_API_KEY")
    # )


# Q&A endpoint for learning path questions
from fastapi import Body

# @app.post("/learning-path-qa")
# def learning_path_qa(
#     question: str = Body(..., embed=True),
#     user_context: str = Body(None, embed=True)
# ):
#     answer = qa_agent.answer(question, user_context)
#     return {"answer": answer}


@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        # Run qa_bot workflow
        thread_id = request.thread_id or str(uuid.uuid4())
        result = qa_bot_app.invoke({
            "query": request.query,
            "web_results": [],
            "db_results": [],
            "tools_to_call": [],
            "user_id": request.user_id,
            "thread_id": thread_id,
            "conversation_history": []
        })
        return {"response": result["final_answer"], "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}
