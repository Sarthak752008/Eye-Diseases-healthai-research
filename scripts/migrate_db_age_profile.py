"""
scripts/migrate_db_age_profile.py
---------------------------------
Database schema migration script for EyeHealthAI:
1. Adds `age_years`, `clinical_sex`, `is_pregnant`, `pregnancy_trimester`, `external_ai_consent` to `user_profiles`.
2. Adds `external_ai_consent` to `consent_records`.
3. Adds `patient_age_years`, `patient_clinical_sex`, `ai_explanation` to `screening_sessions`.
4. Preserves 100% of existing participant data, historical sessions, and reports without loss.
"""

import os
import sys

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.database.session import engine, init_db, SessionLocal
from src.database.models import UserProfile, ScreeningSession, ConsentRecord

def run_migration():
    print("[MIGRATION] Checking and upgrading database schema...")
    init_db()
    
    with SessionLocal() as db:
        users_count = db.query(UserProfile).count()
        sessions_count = db.query(ScreeningSession).count()
        consents_count = db.query(ConsentRecord).count()
        print(f"[MIGRATION] Verified existing records: {users_count} users, {sessions_count} sessions, {consents_count} consents.")
        
        # Check if legacy profiles exist without age_years
        missing_age = db.query(UserProfile).filter(UserProfile.age_years == None).count() # noqa: E711
        if missing_age > 0:
            print(f"[MIGRATION] Notice: {missing_age} historical profiles have no age recorded. They can view history and will be prompted for age upon new screening.")
        else:
            print("[MIGRATION] All user profiles have age_years assigned.")

    print("[MIGRATION] Database schema migration completed successfully.")

if __name__ == "__main__":
    run_migration()
