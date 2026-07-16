import sqlite3

DB_NAME = "reviews.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        movie_name TEXT,
        user_rating REAL,
        review TEXT,
        sentiment TEXT,
        confidence TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_review(movie_name, rating, review, sentiment, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO reviews
    (movie_name,user_rating,review,sentiment,confidence)
    VALUES (?,?,?,?,?)
    """, (
        movie_name,
        rating,
        review,
        sentiment,
        confidence
    ))

    conn.commit()
    conn.close()