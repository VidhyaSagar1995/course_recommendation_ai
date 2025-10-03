# Course Recommendation AI

This project is a student course recommendation system built with FastAPI, LangChain, and various AI tools to provide personalized course suggestions based on student profiles or natural language queries. It includes a web-based interface and a backend API for interaction.

### 1. UI nterface and backend server or CLI for Interaction
- **Web Interface**: Accessible at [https://vidhyasagar1995.github.io/course_recommendation_ai/](https://vidhyasagar1995.github.io/course_recommendation_ai/). Users can interact via a chatbot icon in the bottom right corner.
- **Backend CLI**: Run the API locally after setup (see below) and interact using HTTP requests (e.g., via Postman or curl).

### 2. Sample Input/Output Examples
- **Input (Structured Profile)**:
  ```json
  {
    "interests": ["Machine Learning", "Python"],
    "skill_level": "Intermediate",
    "gials": "to become AI engineer from devops engineer"
  }
  ```
  - **Output**:
    ```json
    {
      "recommendations": [
        {
          "title": "Machine Learning Basics",
          "provider": "Coursera",
          "url": "https://coursera.org/ml",
          "description": "Machine Learning fundamentals."
        }
      ]
    }
    ```

- **Input (Paragraph)**:
  ```json
  {
    "profile_paragraph": "I am a Python developer now, I want to become an AI engineer. Is it possible?",
    "name": "test_user"
  }
  ```
  - **Output** (via Chatbot):
    ```
    Yes, it's absolutely possible to transition from a Python developer to an AI engineer! Python is a strong foundation for AI/ML. Consider these courses:
    - [Machine Learning Basics](https://coursera.org/ml) (Coursera, Beginner, 4 weeks)
    - [Deep Learning Specialization](https://coursera.org/deeplearning) (Coursera, Intermediate, 3 months)
    Let me know if you'd like more details!
    ```

- **Input (Greeting)**:
  ```json
  {
    "query": "hi",
    "user_id": "test_user",
  "thread_id": "testthread"
  }
  ```
  - **Output**:
    ```
    Hello! How can I assist you with courses or learning today?
    ```

## Setup Instructions

### Prerequisites
- API keys for Google (`GOOGLE_API_KEY`), Cohere (`COHERE_API_KEY`), and Tavily (`TAVILY_API_KEY`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vidhyasagar1995/course_recommendation_ai.git
   cd course_recommendation_ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables (create a `.env` file in `app/`):
   ```
   GOOGLE_API_KEY=your_google_api_key
   COHERE_API_KEY=your_cohere_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

### Running the Backend
- Start the FastAPI server:
  ```bash
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```
- Access the API docs at `http://localhost:8000/docs` to test endpoints like `/recommend` or `/query`.

### Using the Web Interface
- Visit [https://vidhyasagar1995.github.io/course_recommendation_ai/](https://vidhyasagar1995.github.io/course_recommendation_ai/).
- Click the chatbot icon (bottom right) to interact with the system using natural language.

### Deployment
- The backend can be deployed on platforms like Render (free tier) with the same `uvicorn` command in the start script.
- Ensure `courses.json` and `app/` are included in the repository root for deployment.

## Contributing
Feel free to fork this repository, submit issues, or pull requests for enhancements!

## License
[MIT License] - See the LICENSE file for details.
