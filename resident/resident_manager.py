import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "safety.db")


def add_resident(
    resident_id,
    name,
    age,
    disease,
    medication,
    allergy,
    emergency_contact,
    fall_history
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO residents (
            resident_id,
            name,
            age,
            disease,
            medication,
            allergy,
            emergency_contact,
            fall_history
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        resident_id,
        name,
        age,
        disease,
        medication,
        allergy,
        emergency_contact,
        fall_history
    ))

    conn.commit()
    conn.close()

    print("Resident added successfully.")


if __name__ == "__main__":
    add_resident(
        "R001",
        "Test User",
        80,
        "High blood pressure",
        "Medicine A",
        "None",
        "090-0000-0000",
        "No previous fall"
    )