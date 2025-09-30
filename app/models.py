from pydantic import BaseModel
from typing import List, Optional


class ParagraphProfile(BaseModel):
    name: str
    profile_paragraph: str

class StudentProfile(BaseModel):
    name: str
    background: Optional[str] = None
    interests: List[str]
    previous_courses: Optional[List[str]] = None
    goals: Optional[str] = None
    skill_levels: Optional[dict] = None  # e.g., {"programming": "intermediate", "mathematics": "advanced"}

class Course(BaseModel):
    id: str
    title: str
    description: str
    skill_level: str
    tags: List[str]
    duration: str
    url: str
    provider: Optional[str]

# Title, provider, description, skill level, tags (e.g., “data-science”, 
# “beginner”)

class RecommendationResponse(BaseModel):
    user_id: str
    recommended_courses: List[Course]


class Feedback(BaseModel):
    user_id: str          # identify the student
    course_id: str        # which course was liked/disliked
    feedback: str         # "like" or "dislike"


class QueryRequest(BaseModel):
    query: str
    user_id: str
    thread_id: Optional[str] = None  # For future use, if needed