# --- Feedback DB (SQLite) ---
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'feedback.db')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'courses.json')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_feedback_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            user_id TEXT NOT NULL,
            course_id TEXT NOT NULL,
            feedback TEXT NOT NULL,
            PRIMARY KEY (user_id, course_id)
        )
    ''')
    conn.commit()
    conn.close()

def init_courses_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            skill_level TEXT NOT NULL,
            tags TEXT NOT NULL,
            duration TEXT NOT NULL,
            url TEXT NOT NULL,
            provider TEXT
        )
    ''')
    conn.commit()
    conn.close()

def dump_courses_to_db():
    print("dumping courses to db")
    # Only insert if table is empty
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM courses').fetchone()[0]
    if count == 0:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            raw_courses = json.load(f)
        for course in raw_courses:
            conn.execute('''
                INSERT INTO courses (id, title, description, skill_level, tags, duration, url, provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                course['id'],
                course['title'],
                course['description'],
                course['skill_level'],
                json.dumps(course['tags']),
                course['duration'],
                course['url'],
                course.get('provider')
            ))
        conn.commit()
    conn.close()


# Initialize SQLite for conversation persistence
def init_convo_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (thread_id TEXT, user_id TEXT, query TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


def init_db():
    init_courses_db()
    dump_courses_to_db()
    init_feedback_db()
    init_convo_db()

def save_feedback(user_id, course_id, feedback):
    conn = get_db_connection()
    conn.execute('''
        INSERT OR REPLACE INTO feedback (user_id, course_id, feedback)
        VALUES (?, ?, ?)
    ''', (user_id, course_id, feedback))
    conn.commit()
    conn.close()

def get_user_feedback(user_id):
    conn = get_db_connection()
    rows = conn.execute('SELECT course_id, feedback FROM feedback WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return {row['course_id']: row['feedback'] for row in rows}

def get_all_feedback():
    conn = get_db_connection()
    rows = conn.execute('SELECT user_id, course_id, feedback FROM feedback').fetchall()
    conn.close()
    result = {}
    for row in rows:
        result.setdefault(row['user_id'], {})[row['course_id']] = row['feedback']
    return result

import json
import os
from app.models import Course




def get_all_courses():
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM courses').fetchall()
    conn.close()
    from app.models import Course
    courses = []
    for row in rows:
        course_dict = dict(row)
        course_dict['tags'] = json.loads(course_dict['tags'])
        courses.append(Course(**course_dict))
    return courses
