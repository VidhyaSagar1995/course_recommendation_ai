# --- Feedback DB (SQLite) ---
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'feedback.db')

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

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'courses.json')

def load_courses():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_courses = json.load(f)
    # Convert JSON dicts to Course objects, handling missing fields
    return [
        Course(**course) for course in raw_courses
    ]

def get_all_courses():
    return load_courses()
