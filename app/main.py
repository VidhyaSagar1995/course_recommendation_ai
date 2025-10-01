import uuid
from typing import Union
from fastapi import FastAPI, HTTPException
from app.recommender import recommend_courses, get_recommender_resources
from app.database import save_feedback

from app.models import StudentProfile, RecommendationResponse, ParagraphProfile, Feedback, QueryRequest
from app.qa_bot import app as qa_bot_app
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
@app.on_event("startup")
def startup_event():
    init_db()
    get_recommender_resources()


@app.post("/query")
async def process_query(request: QueryRequest):
    try:
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
