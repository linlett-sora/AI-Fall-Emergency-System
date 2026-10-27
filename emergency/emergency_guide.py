def get_rescue_guidance(alert_type):
    """Provides rescue steps and practical examples (eg) based on the alert type."""
    
    guidelines = {
        "FALL_DETECTED": {
            "steps": [
                "1. Check the resident's level of consciousness (Are they responsive?).",
                "2. DO NOT move or force the resident to stand up immediately.",
                "3. Check for open airways and ensure they are breathing normally.",
                "4. Look for severe external injuries or signs of bleeding.",
                "5. Call emergency medical services (e.g., 119/192) if severe pain persists."
            ],
        },
        "NO_MOVEMENT": {
            "steps": [
                "1. Check the resident's breathing by observing nose and chest movements for 5 seconds.",
                "2. Try to feel for a pulse at the wrist or neck.",
                "3. If they are unresponsive and not breathing, start CPR immediately.",
                "4. Shout for help and request someone to call the emergency response team."
            ],
        }
    }
    
    return guidelines.get(alert_type, {
        "steps": ["Assess the situation carefully and provide immediate assistance as needed."],
    })


def display_emergency_screen(resident_info, alert_type):
    """Displays the resident profile, specific rescue guidelines, and examples on the screen."""
    print("\n" + "="*65)
    print(f"🚨 [RESCUE MODE ACTIVE] - STATUS: {alert_type} 🚨")
    print("="*65)
    
    print(f"👤 Resident Name : {resident_info.get('name', 'Unknown')}")
    print(f"🏥 Medical Profile: {resident_info.get('disease', 'None')}")
    print(f"💊 Current Meds   : {resident_info.get('medication', 'None')}")
    print(f"📞 Emergency Contact: {resident_info.get('emergency_contact', 'None')}")
    
    print("\n📋 [CRITICAL FIRST-AID INSTRUCTIONS]")
    data = get_rescue_guidance(alert_type)
    
    # Printing steps row by row
    for step in data["steps"]:
        print(step)
        
    print("="*65 + "\n")


if __name__ == "__main__":
    # Sample data for testing local execution
    sample_resident = {
        "name": "U Ba",
        "disease": "High blood pressure",
        "medication": "Amlodipine",
        "emergency_contact": "09-12345678"
    }
    
    # Testing the output layout for Fall Detection
    display_emergency_screen(sample_resident, "FALL_DETECTED")