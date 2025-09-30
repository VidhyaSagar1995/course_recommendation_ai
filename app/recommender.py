# from typing import Union
# # AI logic for course recommendation using LangChain/LangGraph
# from app.models import StudentProfile, RecommendationResponse, ParagraphProfile, Course
# from app.database import get_all_courses

# # Placeholder for LangChain/LangGraph logic
# # Replace with actual AI pipeline as needed
# from sentence_transformers import SentenceTransformer
# import numpy as np
import faiss

# user_feedback = {}

# print("Starting embedding the models")
# # Cache model, courses, embeddings, and FAISS index
# model = SentenceTransformer('all-MiniLM-L6-v2')
# all_courses = get_all_courses()
# course_texts = [f"{c.title} {c.description} {' '.join(c.tags)}" for c in all_courses]
# course_embeddings = model.encode(course_texts, convert_to_numpy=True)
# dim = course_embeddings.shape[1]
# faiss_index = faiss.IndexFlatL2(dim)
# faiss_index.add(course_embeddings)
# print("Embedded courses successfully")


# def adjust_user_embedding(user_id: str, user_embedding: np.ndarray):
#     if user_id not in user_feedback:
#         return user_embedding

#     liked = []
#     disliked = []

#     for cid, fb in user_feedback[user_id].items():
#         course = next((c for c in all_courses if c.id == cid), None)
#         if not course:
#             continue
#         text = f"{course.title} {course.description} {' '.join(course.tags)}"
#         c_emb = model.encode([text], convert_to_numpy=True)
#         if fb == "like":
#             liked.append(c_emb)
#         elif fb == "dislike":
#             disliked.append(c_emb)

#     if liked:
#         user_embedding = user_embedding + 0.2 * np.mean(liked, axis=0)
#     if disliked:
#         user_embedding = user_embedding - 0.2 * np.mean(disliked, axis=0)

#     return user_embedding


# def recommend_courses(student: Union[StudentProfile, ParagraphProfile]) -> RecommendationResponse:
#     """
#     Recommend courses using cached sentence embeddings and FAISS index.
#     """
#     if isinstance(student, StudentProfile):
#         profile_text = f"{student.background or ''} {' '.join(student.interests)} {student.goals or ''}"
#         previous_courses = student.previous_courses
#     elif isinstance (student, ParagraphProfile):
#         profile_text = student.profile_paragraph
#         previous_courses = []
#     else:
#         previous_courses = []
#         return []
#     user_embedding = model.encode([profile_text], convert_to_numpy=True)
#     # user_embedding = model.encode([profile_text], convert_to_numpy=True)
#     user_embedding = adjust_user_embedding(student.name, user_embedding)  # 👈 feedback applied

#     D, I = faiss_index.search(user_embedding, k=5)
#     recommended = [all_courses[i] for i in I[0] if all_courses[i].id not in previous_courses]
#     for i in recommended:
#         print(type(i), i.id)
#     return RecommendationResponse(recommended_courses=recommended)




from typing import Union
from app.models import StudentProfile, RecommendationResponse, ParagraphProfile, Course
from app.database import get_all_courses, get_user_feedback, get_all_feedback
from sentence_transformers import SentenceTransformer
import numpy as np



# Lazy initialization for model, courses, embeddings, and FAISS index
_model = None
_all_courses = None
_course_texts = None
_course_embeddings = None
_faiss_index = None

def get_recommender_resources():
    global _model, _all_courses, _course_texts, _course_embeddings, _faiss_index
    if _model is None:
        print("Starting embedding the models")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    if _all_courses is None:
        _all_courses = get_all_courses()
    if _course_texts is None:
        _course_texts = [f"{c.title} {c.description} {' '.join(c.tags)}" for c in _all_courses]
    if _course_embeddings is None:
        _course_embeddings = _model.encode(_course_texts, convert_to_numpy=True)
    if _faiss_index is None:
        dim = _course_embeddings.shape[1]
        _faiss_index = faiss.IndexFlatL2(dim)
        _faiss_index.add(_course_embeddings)
        print("Embedded courses successfully")
    return _model, _all_courses, _faiss_index

def adjust_user_embedding(user_id: str, user_embedding: np.ndarray):
    model, all_courses, _ = get_recommender_resources()
    feedback_dict = get_user_feedback(user_id)
    if not feedback_dict:
        return user_embedding

    liked = []
    disliked = []

    for cid, fb in feedback_dict.items():
        course = next((c for c in all_courses if c.id == cid), None)
        if not course:
            continue
        text = f"{course.title} {course.description} {' '.join(course.tags)}"
        c_emb = model.encode([text], convert_to_numpy=True)
        if fb == "like":
            liked.append(c_emb)
        elif fb == "dislike":
            disliked.append(c_emb)

    if liked:
        user_embedding = user_embedding + 0.2 * np.mean(liked, axis=0)
    if disliked:
        user_embedding = user_embedding - 0.2 * np.mean(disliked, axis=0)

    return user_embedding

def recommend_courses(student: Union[StudentProfile, ParagraphProfile]) -> RecommendationResponse:
    """
    Recommend courses using cached sentence embeddings and FAISS index, excluding disliked courses.
    """
    print("feedbacks", get_user_feedback(student.name))
    model, all_courses, faiss_index = get_recommender_resources()
    if isinstance(student, StudentProfile):
        profile_text = f"{student.background or ''} {' '.join(student.interests)} {student.goals or ''}"
        previous_courses = student.previous_courses
    else:
        profile_text = student.profile_paragraph
        previous_courses = []

    user_embedding = model.encode([profile_text], convert_to_numpy=True)
    user_embedding = adjust_user_embedding(student.name, user_embedding)

    # Get top 10 candidates to ensure enough results after filtering
    D, I = faiss_index.search(user_embedding, k=10)
    
    # Filter out previous courses and disliked courses
    disliked_courses = {
        course_id for course_id, feedback in get_user_feedback(student.name).items() 
        if feedback == 'dislike'
    }
    recommended = [
        all_courses[i] for i in I[0]
        if all_courses[i].id not in previous_courses and all_courses[i].id not in disliked_courses
    ]
    
    # Limit to top 5 recommendations
    recommended = recommended[:5]
    return RecommendationResponse(user_id=student.name, recommended_courses=recommended)