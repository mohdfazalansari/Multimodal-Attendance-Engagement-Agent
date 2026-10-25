import asyncio
import datetime
import numpy as np
import os
import shutil
from sqlalchemy.future import select
from app.core.database import AsyncSessionLocal, engine, Base
from app.core.init_db import init_database
from app.models.db_models import Student, Session, AttendanceRecord, EngagementWindow, TeachingSuggestion, ConsentLog, Course
from app.core.config import settings

async def seed_full_classroom_data():
    # Recreate tables cleanly
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Clean embeddings dir
    for f in settings.EMBEDDINGS_DIR.glob("*.npy"):
        try:
            os.remove(f)
        except OSError:
            pass

    async with AsyncSessionLocal() as db:
        print("Seeding professional classroom demo data...")

        students_data = [
            ("CS-2024-001", "Emma Watson", "Computer Science 101", "emma.watson@university.edu", True),
            ("CS-2024-002", "Liam Smith", "Computer Science 101", "liam.smith@university.edu", True),
            ("CS-2024-003", "Sophia Garcia", "Computer Science 101", "sophia.g@university.edu", True),
            ("CS-2024-004", "Noah Patel", "Computer Science 101", "noah.patel@university.edu", True),
            ("CS-2024-005", "Ava Chen", "Computer Science 101", "ava.chen@university.edu", True),
            ("CS-2024-006", "Lucas Vance", "Computer Science 101", "lucas.v@university.edu", True),
            ("CS-2024-007", "Maya Lin", "Computer Science 101", "maya.lin@university.edu", True),
            ("CS-2024-008", "Ethan Wright", "Computer Science 101", "ethan.w@university.edu", False), # Demonstrates unconsented
        ]

        created_students = []
        now = datetime.datetime.now(datetime.UTC)

        for sid, name, course, email, consented in students_data:
            s = Student(
                student_id=sid,
                full_name=name,
                course_name=course,
                email=email,
                has_consented=consented,
                consent_date=now - datetime.timedelta(days=14) if consented else None,
                embedding_path=None
            )
            db.add(s)
            created_students.append(s)

        await db.flush()

        for s in created_students:
            log = ConsentLog(
                student_id=s.id,
                consent_action="GRANTED" if s.has_consented else "REVOKED",
                notes="Explicit biometric consent recorded" if s.has_consented else "Student opted out of biometric storage",
                timestamp=now - datetime.timedelta(days=14)
            )
            db.add(log)

            if s.has_consented:
                vec = np.random.randn(512).astype(np.float32)
                vec /= np.linalg.norm(vec)
                emb_path = settings.EMBEDDINGS_DIR / f"student_{s.id}_{s.student_id}.npy"
                np.save(str(emb_path), vec)
                s.embedding_path = str(emb_path)

        # Session 1: Completed CS 101 Lecture 1
        sess1 = Session(
            session_uid="SESS-CS101-01",
            course_title="Computer Science 101: Intro to Algorithms",
            instructor_name="Dr. Ada Lovelace",
            room_number="Turing Hall 102",
            target_class="Computer Science 101",
            expected_duration_minutes=50,
            status="COMPLETED",
            source_type="LIVE",
            start_time=now - datetime.timedelta(days=3, hours=2),
            end_time=now - datetime.timedelta(days=3, hours=1, minutes=10),
            actual_duration_seconds=3000.0
        )
        db.add(sess1)

        # Session 2: Completed CS 101 Lecture 2
        sess2 = Session(
            session_uid="SESS-CS101-02",
            course_title="Computer Science 101: Graph Traversal & Search",
            instructor_name="Dr. Ada Lovelace",
            room_number="Turing Hall 102",
            target_class="Computer Science 101",
            expected_duration_minutes=50,
            status="COMPLETED",
            source_type="LIVE",
            start_time=now - datetime.timedelta(days=1, hours=3),
            end_time=now - datetime.timedelta(days=1, hours=2, minutes=10),
            actual_duration_seconds=3000.0
        )
        db.add(sess2)

        # Session 3: Scheduled / Video Ingestion target
        sess3 = Session(
            session_uid="SESS-CS101-03",
            course_title="Computer Science 101: Dynamic Programming Analysis",
            instructor_name="Dr. Ada Lovelace",
            room_number="Lecture Hall 3",
            target_class="Computer Science 101",
            expected_duration_minutes=45,
            status="SCHEDULED",
            source_type="VIDEO_UPLOAD",
            start_time=None,
            end_time=None,
            actual_duration_seconds=0.0
        )
        db.add(sess3)

        await db.flush()

        # Session 1 Attendance Records
        statuses1 = ["PRESENT", "PRESENT", "LATE", "PRESENT", "PRESENT", "PRESENT", "PRESENT", "UNVERIFIED"]
        notes1 = [
            "Sustained presence confirmed (38 observations, 92% match)",
            "Sustained presence confirmed (41 observations, 89% match)",
            "Arrived at minute 14. Sustained presence confirmed (27 observations)",
            "Sustained presence confirmed (35 observations, 94% match)",
            "Sustained presence confirmed (33 observations, 91% match)",
            "Sustained presence confirmed (39 observations, 88% match)",
            "Sustained presence confirmed (36 observations, 90% match)",
            "Consent revoked: face embedding disabled; logged as unverified presence"
        ]
        for idx, s in enumerate(created_students):
            att = AttendanceRecord(
                session_id=sess1.id,
                student_id=s.id,
                detected_name=s.full_name,
                status=statuses1[idx],
                first_seen_seconds=25.0 if statuses1[idx] != "LATE" else 840.0,
                last_seen_seconds=2950.0,
                sustained_duration_seconds=2700.0 if statuses1[idx] != "LATE" else 1900.0,
                observation_count=35 if statuses1[idx] != "UNVERIFIED" else 12,
                avg_confidence=0.91 if statuses1[idx] != "UNVERIFIED" else 0.45,
                is_manual_override=False,
                notes=notes1[idx]
            )
            db.add(att)

        # Session 2 Attendance Records
        statuses2 = ["PRESENT", "PRESENT", "PRESENT", "LATE", "PRESENT", "LEFT_EARLY", "PRESENT", "UNVERIFIED"]
        notes2 = [
            "Sustained presence confirmed (42 observations, 93% match)",
            "Sustained presence confirmed (39 observations, 91% match)",
            "Sustained presence confirmed (44 observations, 95% match)",
            "Arrived at minute 12 (sustained 28 observations)",
            "Sustained presence confirmed (37 observations, 89% match)",
            "Departed at minute 32; sustained duration below 60% threshold",
            "Sustained presence confirmed (40 observations, 92% match)",
            "Consent revoked: manual instructor check recommended"
        ]
        for idx, s in enumerate(created_students):
            att = AttendanceRecord(
                session_id=sess2.id,
                student_id=s.id,
                detected_name=s.full_name,
                status=statuses2[idx],
                first_seen_seconds=20.0 if statuses2[idx] != "LATE" else 720.0,
                last_seen_seconds=1920.0 if statuses2[idx] == "LEFT_EARLY" else 2980.0,
                sustained_duration_seconds=1850.0 if statuses2[idx] == "LEFT_EARLY" else 2800.0,
                observation_count=38 if statuses2[idx] != "LEFT_EARLY" else 22,
                avg_confidence=0.92,
                is_manual_override=False,
                notes=notes2[idx]
            )
            db.add(att)

        # Engagement Windows for Session 1 (12 windows x 30s)
        # Demonstrates natural pedagogical arc: high start, interactive discussion peak, steady wrap
        scores = [82.0, 85.0, 78.0, 74.0, 88.0, 92.0, 86.0, 79.0, 75.0, 81.0, 84.0, 80.0]
        poses = [0.86, 0.88, 0.80, 0.76, 0.90, 0.94, 0.89, 0.82, 0.77, 0.84, 0.87, 0.83]
        motions = [0.24, 0.28, 0.32, 0.30, 0.48, 0.52, 0.44, 0.35, 0.29, 0.31, 0.33, 0.25]
        speeches = [0.35, 0.42, 0.28, 0.22, 0.65, 0.70, 0.58, 0.34, 0.25, 0.38, 0.40, 0.30]

        for w_idx in range(len(scores)):
            win = EngagementWindow(
                session_id=sess1.id,
                window_index=w_idx,
                start_seconds=w_idx * 30.0,
                end_seconds=(w_idx + 1) * 30.0,
                person_count_avg=7.2,
                forward_facing_ratio=poses[w_idx],
                movement_intensity=motions[w_idx],
                prolonged_inactivity_ratio=0.08,
                speech_active_ratio=speeches[w_idx],
                interaction_activity_score=scores[w_idx],
                is_audio_available=True,
                is_video_available=True,
                notes="Multi-sensor observable signal aggregation"
            )
            db.add(win)

        # Engagement Windows for Session 2
        scores2 = [76.0, 79.0, 81.0, 85.0, 87.0, 84.0, 78.0, 72.0, 68.0, 74.0, 80.0, 82.0]
        for w_idx in range(len(scores2)):
            win = EngagementWindow(
                session_id=sess2.id,
                window_index=w_idx,
                start_seconds=w_idx * 30.0,
                end_seconds=(w_idx + 1) * 30.0,
                person_count_avg=6.9,
                forward_facing_ratio=0.81,
                movement_intensity=0.33,
                prolonged_inactivity_ratio=0.12,
                speech_active_ratio=0.31,
                interaction_activity_score=scores2[w_idx],
                is_audio_available=True,
                is_video_available=True,
                notes="Multi-sensor observable signal aggregation"
            )
            db.add(win)

        # Teaching Suggestions for Session 1
        sugs1 = [
            TeachingSuggestion(
                session_id=sess1.id,
                suggestion_type="INTERACTION_PEAK",
                title="Active Discussion Segment Observed",
                description="Classroom behavioral signals reflected elevated collaborative interaction between minutes 2:00 and 3:30.",
                evidence="VAD speech activity reached 70% accompanied by increased physical movement and multi-person orientation.",
                urgency="LOW",
                generated_by="RULE_BASED",
                acknowledged=True
            ),
            TeachingSuggestion(
                session_id=sess1.id,
                suggestion_type="STEADY_FLOW",
                title="Sustained Forward Attention Pattern",
                description="85% of detected students maintained frontal head-pose orientation toward the primary presentation area.",
                evidence="solvePnP yaw angles remained within +/- 18 degrees across 9 consecutive 30-second windows.",
                urgency="LOW",
                generated_by="RULE_BASED",
                acknowledged=False
            ),
            TeachingSuggestion(
                session_id=sess1.id,
                suggestion_type="ATTENDANCE_VERIFICATION",
                title="Unconsented Attendee Detected in Cohort",
                description="Ethan Wright has opted out of facial biometric storage. Sustained presence detected via tracking but identity unverified.",
                evidence="Biometric consent status is FALSE. Privacy safeguards prevent feature vector extraction.",
                urgency="MEDIUM",
                generated_by="RULE_BASED",
                acknowledged=False
            )
        ]
        for sug in sugs1:
            db.add(sug)

        await db.commit()
        print("Classroom demo dataset successfully populated with realistic telemetry!")

if __name__ == "__main__":
    asyncio.run(seed_full_classroom_data())
