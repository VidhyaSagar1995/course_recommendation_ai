# Student Course Recommendation AI

A FastAPI-based AI software to recommend courses to students based on their profile, interests, and previous courses. Uses LangChain and LangGraph for advanced recommendation logic.

## Features
- Student profile input
- Course database
- AI-powered recommendations
- REST API endpoints

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   uvicorn app.main:app --reload
   ```
3. Access API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## File Structure
- `app/main.py`: FastAPI entry point
- `app/models.py`: Data models
- `app/database.py`: Course data
- `app/recommender.py`: Recommendation logic
- `app/utils.py`: Utilities

## Next Steps
- Integrate real database
- Enhance AI logic with LangChain/LangGraph
- Add authentication and more endpoints
