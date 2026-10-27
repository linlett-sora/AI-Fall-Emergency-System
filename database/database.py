import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "safety.db")


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS residents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            disease TEXT,
            medication TEXT,
            allergy TEXT,
            emergency_contact TEXT,
            fall_history TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emergency_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id TEXT,
            event_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'ACTIVE',
            FOREIGN KEY (resident_id) REFERENCES residents(resident_id)
        )
    """)
        

    conn.commit()
    conn.close()

    print(f"Database created successfully at: {DB_PATH}")


if __name__ == "__main__":
    create_database()