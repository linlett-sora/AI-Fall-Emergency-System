import sqlite3
import os
import winsound
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "safety.db")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_system_email@gmail.com"
SENDER_PASSWORD = "your_app_password"

def send_email_notification(guardian_email, resident_name, alert_type, timestamp, resident_info):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = guardian_email
        msg['Subject'] = f"[🚨 EMERGENCY ALERT] {alert_type} - {resident_name}"
        body = f"""
⚠️ Emergency Alert (Emergency Alert)

resident name: {resident_name}
alert type: {alert_type}
timestamp: {timestamp}

[resident_info]
disease: {resident_info['disease']}
medication: {resident_info['medication']}
allergy: {resident_info['allergy']}
emergency contact: {resident_info['emergency_contact']}

        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, guardian_email, msg.as_string())
        server.quit()
        print(f" [✉️] Emergency email sent to guardian ({guardian_email}) successfully.")
    except Exception as e:
        print(f" [❌] Failed to send email: {e}")

def trigger_emergency_alert(resident_id, alert_type="FALL_DETECTED"):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[🚨 LIVE ALERT] {alert_type} at {current_time} for ID: {resident_id}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, disease, medication, allergy, emergency_contact FROM residents WHERE resident_id = ?", (resident_id,))
    resident = cursor.fetchone()

    if not resident:
        print(f" [❌] Error: Resident ID {resident_id} not found in database.")
        conn.close()
        return
    resident_info = {
        "name": resident[0],
        "disease": resident[1],
        "medication": resident[2],
        "allergy": resident[3],
        "emergency_contact": resident[4]
    }

    try:
        cursor.execute("""
            INSERT INTO emergency_events (resident_id, event_type, timestamp, status)
            VALUES (?, ?, ?, 'ACTIVE')
        """, (resident_id, alert_type, current_time))
        conn.commit()
        print(f" [✓] Event data auto-saved to database at {current_time}.")
    except sqlite3.Error as e:
        print(f" [❌] Database Auto-save Error: {e}")
    finally:
        conn.close()
    
    for _ in range(3):
        winsound.Beep(2000, 400)
    guardian_email = "guardian_email_address@gmail.com" 
    send_email_notification(guardian_email, resident_info['name'], alert_type, current_time, resident_info)


if __name__ == "__main__":
    trigger_emergency_alert("R001", "FALL_DETECTED")


# for Role 2 to add code
#from alert.alarm import trigger_emergency_alert
#if fall_is_detected == True:
#    trigger_emergency_alert("R001", "FALL_DETECTED")