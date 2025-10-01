import faiss
from typing import Union
from app.models import StudentProfile, RecommendationResponse, ParagraphProfile, Course
from app.database import get_all_courses, get_user_feedback, get_all_feedback
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
# import os
# from langchain_cohere import ChatCohere
# from dotenv import load_dotenv
# load_dotenv()

# input_understanding_llm = ChatCohere(model="command-a-03-2025", cohere_api_key=os.getenv("COHERE_API_KEY"))




# Lazy initialization for model, courses, embeddings, and FAISS index
_model = None
_all_courses = None
_course_texts = None
_course_embeddings = None
_faiss_index = None


def preprocess_input(student: Union[StudentProfile, ParagraphProfile]) -> tuple[str, str]:
    """
    Preprocess user input to generate a focused learning query and explanation.
    For StudentProfile, prioritize goals, adjust for skill_levels, and exclude previous_courses.
    For ParagraphProfile, parse structured string or use LLM for unstructured input.
    """
    if isinstance(student, StudentProfile):
        # Extract fields
        background = student.background or ""
        interests = " ".join(student.interests) if student.interests else ""
        goals = student.goals or ""
        previous_courses = " ".join(student.previous_courses) if student.previous_courses else ""
        skill_levels = student.skill_levels or {}

        # Construct query with weighted fields
        query_parts = []
        explanation_parts = []

        # Prioritize goals (weight: 2x to emphasize)
        if goals:
            query_parts.append(goals + " " + goals)
            if "react" in goals.lower():
                query_parts.append("react javascript frontend")
            elif "devops" in goals.lower():
                query_parts.append("ci-cd kubernetes docker")
            explanation_parts.append("Prioritized courses matching your goal.")

        # Add interests (weight: 0.5)
        if interests:
            query_parts.append(interests)
            explanation_parts.append("Included courses aligned with your interests.")

        # Add background as context (weight: 0.3)
        if background:
            query_parts.append(background)
            explanation_parts.append("Considered your background for relevant courses.")

        # Handle skill_levels: adjust query based on proficiency
        if skill_levels:
            skill_query = []
            for skill, level in skill_levels.items():
                if level == "beginner":
                    skill_query.append(f"beginner {skill}")
                    explanation_parts.append(f"Recommended beginner-level {skill} courses due to your skill level.")
                elif level == "intermediate":
                    skill_query.append(f"intermediate {skill}")
                    explanation_parts.append(f"Recommended intermediate-level {skill} courses.")
                elif level == "advanced":
                    skill_query.append(f"advanced {skill}")
                    explanation_parts.append(f"Deprioritized {skill} courses due to your advanced skill.")
            query_parts.append(" ".join(skill_query))

        # Add bridging term if goals conflict with interests/background
        if goals and (interests or background):
            goal_keywords = set(goals.lower().split())
            interest_keywords = set(interests.lower().split())
            if len(goal_keywords & interest_keywords) < len(goal_keywords) * 0.5:
                query_parts.append("bridge mlops")
                explanation_parts.append("Added MLOps to bridge your background/interests to your goal.")

        # Exclude previous courses
        if previous_courses:
            explanation_parts.append("Excluded courses similar to your previous ones.")

        query = " ".join(query_parts)
        explanation = " ".join(explanation_parts) or "Generated recommendations based on your profile."

    else:  # ParagraphProfile
        # Try parsing structured string format
        structured_pattern = r"career goal is ([^,]+) with interests ([^ ]+) and my current background is ([^ ]+) i have following skills ([^ ]+) i already pursued ([^ ]+)"
        match = re.match(structured_pattern, student.profile_paragraph, re.IGNORECASE)
        
        if match:
            goal = match.group(1).strip()
            interests = match.group(2).strip().replace(",", " ")
            background = match.group(3).strip()
            skills_raw = match.group(4).strip()
            previous_courses = match.group(5).strip().replace(",", " ")

            # Parse skills (e.g., "nlp with proficiency beginner python with proficiency advanced")
            skill_pairs = re.findall(r"(\w+) with proficiency (\w+)", skills_raw, re.IGNORECASE)
            skill_query = []
            explanation_parts = []
            query_parts = []

            # Prioritize goal
            query_parts.append(goal + " " + goal)
            if "react" in goal.lower():
                query_parts.append("react javascript frontend")
            elif "devops" in goal.lower():
                query_parts.append("ci-cd kubernetes docker")
            explanation_parts.append("Prioritized courses matching your goal.")

            # Add interests
            if interests:
                query_parts.append(interests)
                explanation_parts.append("Included courses aligned with your interests.")

            # Add background
            if background:
                query_parts.append(background)
                explanation_parts.append("Considered your background for relevant courses.")

            # Handle skills
            for skill, level in skill_pairs:
                level = level.lower()
                if level == "beginner":
                    skill_query.append(f"beginner {skill}")
                    explanation_parts.append(f"Recommended beginner-level {skill} courses due to your skill level.")
                elif level == "intermediate":
                    skill_query.append(f"intermediate {skill}")
                    explanation_parts.append(f"Recommended intermediate-level {skill} courses.")
                elif level == "advanced":
                    skill_query.append(f"advanced {skill}")
                    explanation_parts.append(f"Deprioritized {skill} courses due to your advanced skill.")
            query_parts.append(" ".join(skill_query))

            # Exclude previous courses
            if previous_courses:
                explanation_parts.append("Excluded courses similar to your previous ones.")

            # Add bridging term
            if goal and interests:
                goal_keywords = set(goal.lower().split())
                interest_keywords = set(interests.lower().split())
                if len(goal_keywords & interest_keywords) < len(goal_keywords) * 0.5:
                    query_parts.append("bridge mlops")
                    explanation_parts.append("Added MLOps to bridge your background/interests to your goal.")

            query = " ".join(query_parts)
            explanation = " ".join(explanation_parts) or "Generated recommendations based on your profile."

        else:
            query = student.profile_paragraph
            explanation = "Manually extracted goal and background due to LLM processing error."

    return query, None


def old_preprocess_input(student: Union[StudentProfile, ParagraphProfile]) -> tuple[str, str]:
    """
    Preprocess user input (structured or paragraph) to generate a focused learning query and explanation.
    For StudentProfile, use all fields except name. For ParagraphProfile, use LLM to extract intent.
    Adjust query based on skill_levels (e.g., prioritize beginner courses for low skills).
    """
    if isinstance(student, StudentProfile):
        # Extract fields
        background = student.background or ""
        interests = " ".join(student.interests) if student.interests else ""
        goals = student.goals or ""
        previous_courses = " ".join(student.previous_courses) if student.previous_courses else ""
        skill_levels = student.skill_levels or {}

        # Construct query with weighted fields
        query_parts = []
        explanation_parts = []

        # Prioritize goals (weight: 1.0)
        if goals:
            query_parts.append(goals)
            explanation_parts.append("Prioritized courses matching your goal.")

        # Add interests (weight: 0.5)
        if interests:
            query_parts.append(interests)
            explanation_parts.append("Included courses aligned with your interests.")

        # Add background as context (weight: 0.3)
        if background:
            query_parts.append(background)
            explanation_parts.append("Considered your background for relevant courses.")

        # Handle skill_levels: adjust query based on proficiency
        if skill_levels:
            skill_query = []
            for skill, level in skill_levels.items():
                if level == "beginner":
                    skill_query.append(f"beginner {skill}")
                    explanation_parts.append(f"Recommended beginner-level {skill} courses due to your skill level.")
                elif level == "intermediate":
                    skill_query.append(f"intermediate {skill}")
                    explanation_parts.append(f"Recommended intermediate-level {skill} courses.")
                elif level == "advanced":
                    skill_query.append(f"advanced {skill}")
                    explanation_parts.append(f"Deprioritized {skill} courses due to your advanced skill.")
            query_parts.append(" ".join(skill_query))

        # Add bridging term if goals conflict with interests/background
        if goals and (interests or background):
            goal_keywords = set(goals.lower().split())
            interest_keywords = set(interests.lower().split())
            if len(goal_keywords & interest_keywords) < len(goal_keywords) * 0.5:
                query_parts.append("bridge mlops")
                explanation_parts.append("Added MLOps to bridge your background/interests to your goal.")

        # Exclude previous courses
        if previous_courses:
            explanation_parts.append("Excluded courses similar to your previous ones.")

        query = " ".join(query_parts)
        explanation = " ".join(explanation_parts) or "Generated recommendations based on your profile."

    else:  # ParagraphProfile
        prompt = (
            f"User Input: {student.profile_paragraph}\n"
            "Extract the user's learning objective, interests, background, and skills. "
            "Generate a concise query for course recommendation, prioritizing goals and adjusting for skill levels. "
            "Return JSON: {\"query\": \"search query\", \"explanation\": \"reasoning\"}"
        )
        try:
            response = input_understanding_llm.invoke(prompt).content
            data = json.loads(response)
            query = data.get("query", student.profile_paragraph)
            explanation = data.get("explanation", "Extracted intent from paragraph.")
        except Exception as e:
            print("LLM preprocessing error:", e)
            query = student.profile_paragraph
            explanation = "Used raw paragraph due to processing error."
    print("explanation", explanation, "query", query)
    return query, explanation


def preprocess_with_llm(structured_input=None, paragraph_input=None):
    """
    Use LLM to preprocess user input (structured or paragraph) and generate a focused learning objective.
    """
    if structured_input:
        prompt = (
            f"Background: {structured_input.get('background', '')}\n"
            f"Interests: {', '.join(structured_input.get('interests', []))}\n"
            f"Goals: {structured_input.get('goals', '')}\n"
            "Generate a concise learning objective for course recommendation, prioritizing goals and interests."
        )
    elif paragraph_input:
        prompt = (
            f"User Input: {paragraph_input}\n"
            "Extract the user's future learning objective and interests for course recommendation."
        )
    else:
        return ""
    try:
        response = input_understanding_llm.invoke(prompt)
        return response.strip() if isinstance(response, str) else str(response)
    except Exception as e:
        print("LLM preprocessing error:", e)
        # Fallback: return original input
        return paragraph_input if paragraph_input else str(structured_input)

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
    Recommend courses using LLM-preprocessed input, cached sentence embeddings, and FAISS index, excluding disliked courses.
    """
    # print("feedbacks", get_user_feedback(student.name))
    model, all_courses, faiss_index = get_recommender_resources()
    if isinstance(student, StudentProfile):
        structured_input = {
            "background": student.background,
            "interests": student.interests,
            "goals": student.goals
        }
        # processed_text = preprocess_with_llm(structured_input=structured_input)
        previous_courses = student.previous_courses
      
    else:
        # processed_text = preprocess_with_llm(paragraph_input=student.profile_paragraph)
        previous_courses = []
       

    query, explanation = preprocess_input(student)
    user_embedding = model.encode([query], convert_to_numpy=True)
    user_embedding = adjust_user_embedding(student.name, user_embedding)
    # Get top 10 courses to ensure enough results after filtering
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
    return RecommendationResponse(user_id=student.name, 
                                  recommended_courses=recommended,
                                  explanation=explanation)