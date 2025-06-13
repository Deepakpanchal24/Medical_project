import sqlite3
from datetime import datetime

DB_NAME = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_name TEXT,
            question TEXT,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat_history(patient_name, query, answer):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO chat_history (timestamp, patient_name, question, answer) VALUES (?, ?, ?, ?)",
        (timestamp, patient_name, query, answer)
    )
    conn.commit()
    conn.close()

def view_chat_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, patient_name, question, answer FROM chat_history ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "No chat history found."
    formatted = []
    for ts, name, q, a in rows:
        formatted.append(f"**{ts} | {name}**\nQ: {q}\nA: {a}\n")
    return "\n".join(formatted)