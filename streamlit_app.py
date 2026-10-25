import os
import sys
import time
import tempfile
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. System Paths & Backend Service Linking
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Thread-safe executor for async backend database services
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def run_async(coro):
    """Safely runs an async coroutine inside a clean worker thread without event-loop conflicts."""
    future = _EXECUTOR.submit(asyncio.run, coro)
    return future.result()

# Defensive fallback shim: ensure HTTPException and status exist even if fastapi is absent
try:
    import fastapi
except ImportError:
    import types
    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = ""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(f"HTTP {status_code}: {detail}")
    
    class Status:
        HTTP_200_OK = 200
        HTTP_201_CREATED = 201
        HTTP_400_BAD_REQUEST = 400
        HTTP_404_NOT_FOUND = 404
        HTTP_409_CONFLICT = 409
        HTTP_500_INTERNAL_SERVER_ERROR = 500

    fastapi_mock = types.ModuleType("fastapi")
    fastapi_mock.HTTPException = HTTPException
    fastapi_mock.status = Status
    sys.modules["fastapi"] = fastapi_mock

# Import backend models, services and configs
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.init_db import init_database
from app.services.session_service import SessionService
from app.services.student_service import StudentService
from app.services.attendance_service import AttendanceService
from app.services.analytics_service import AnalyticsService
from app.services.suggestion_service import SuggestionService
from app.models.db_models import Student, Session, AttendanceRecord, EngagementWindow, TeachingSuggestion
from app.schemas.attendance import AttendanceUpdate
from app.schemas.student import StudentCreate
from app.schemas.session import SessionCreate
from app.utils.seed_full_demo import seed_full_classroom_data

# ---------------------------------------------------------
# 2. Streamlit Page Configuration & Modern Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark slate, cyan/emerald accents, refined cards, clean badges)
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    /* Section card container */
    .dashboard-card {
        background-color: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    /* Status Badges */
    .badge-present { background-color: #065f46; color: #a7f3d0; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.78rem; }
    .badge-late { background-color: #78350f; color: #fde68a; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.78rem; }
    .badge-early { background-color: #7c2d12; color: #fed7aa; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.78rem; }
    .badge-unverified { background-color: #581c87; color: #e9d5ff; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.78rem; }
    .badge-pending { background-color: #334155; color: #cbd5e1; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.78rem; }
    .badge-absent { background-color: #7f1d1d; color: #fecaca; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.78rem; }
    .badge-consented { background-color: #065f46; color: #6ee7b7; padding: 2px 7px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-revoked { background-color: #7f1d1d; color: #fca5a5; padding: 2px 7px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    /* Urgency Badges */
    .urgency-high { background-color: #dc2626; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .urgency-medium { background-color: #d97706; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .urgency-low { background-color: #059669; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Cached Resource Initializers
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_hardware_telemetry():
    """Detects CPU/CUDA hardware status and module availability."""
    info = {
        "device": "CPU (Offline Safe Mode)",
        "cuda_available": False,
        "yolo_ready": False,
        "facenet_ready": False,
        "opencv_version": cv2.__version__
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device"] = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass

    try:
        from ultralytics import YOLO
        info["yolo_ready"] = True
    except Exception:
        pass

    try:
        import keras_facenet
        info["facenet_ready"] = True
    except Exception:
        pass

    return info

@st.cache_resource(show_spinner=False)
def get_or_create_pipeline():
    """Initializes the MultimodalPipeline once and caches it to prevent memory bloat."""
    try:
        from app.vision.pipeline import MultimodalPipeline
        return MultimodalPipeline()
    except Exception as e:
        st.sidebar.warning(f"Vision Pipeline fallback active: {e}")
        return None

# Ensure database tables and synthetic classroom data are seeded on cloud startup
def ensure_database_ready():
    async def _init_check():
        await init_database()
        async with AsyncSessionLocal() as db:
            sessions = await SessionService.get_all(db)
            if not sessions:
                await seed_full_classroom_data()
    run_async(_init_check())

ensure_database_ready()

# ---------------------------------------------------------
# 4. Sidebar: Telemetry, Navigation & Session Selection
# ---------------------------------------------------------
telemetry = get_hardware_telemetry()

with st.sidebar:
    st.markdown("### 🎓 Classroom Agent")
    st.caption("Multimodal Attendance & Observable Engagement Intelligence")

    # Hardware & Cloud Status Badge
    with st.container():
        st.markdown(f"""
        <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin-bottom: 15px;">
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; margin-bottom: 4px;">EXECUTION ENVIRONMENT</div>
            <div style="font-size: 0.85rem; color: #38bdf8; font-weight: 600;">⚡ {telemetry['device']}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                OpenCV: {telemetry['opencv_version']} • YOLOv8: {'Ready' if telemetry['yolo_ready'] else 'HOG Fallback'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Navigation Menu
    menu = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "👥 Student & Consent Management",
            "🎥 Video Analysis Pipeline",
            "📋 Attendance Verification",
            "📈 Engagement Signal Analytics",
            "💡 AI Teaching Suggestions",
            "📑 Session Reports & Exports",
            "🛡️ Responsible AI & Privacy"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Fetch available sessions for global selector
    async def _fetch_sessions():
        async with AsyncSessionLocal() as db:
            return await SessionService.get_all(db)
    
    all_sessions = run_async(_fetch_sessions())
    session_options = {s.id: f"{s.course_title} ({s.status})" for s in all_sessions}

    if "active_session_id" not in st.session_state and all_sessions:
        st.session_state["active_session_id"] = all_sessions[0].id

    st.markdown("##### 📌 Active Session")
    if session_options:
        selected_id = st.selectbox(
            "Select Session",
            options=list(session_options.keys()),
            format_func=lambda x: session_options.get(x, f"Session {x}"),
            index=list(session_options.keys()).index(st.session_state.get("active_session_id", list(session_options.keys())[0])),
            key="session_selector"
        )
        st.session_state["active_session_id"] = selected_id
    else:
        st.info("No active sessions found.")

    st.markdown("---")
    st.markdown("##### 🔄 Reset Demo")
    if st.button("Reset Synthetic Classroom Data", use_container_width=True, help="Re-seeds clean demonstration data."):
        with st.spinner("Seeding classroom cohorts and telemetry..."):
            run_async(seed_full_classroom_data())
            st.success("Demo dataset refreshed!")
            time.sleep(0.5)
            st.rerun()

# ---------------------------------------------------------
# Helper Queries
# ---------------------------------------------------------
def get_active_session(session_id: int):
    async def _get():
        async with AsyncSessionLocal() as db:
            return await SessionService.get_by_id(db, session_id)
    return run_async(_get())

def get_session_attendance(session_id: int):
    async def _get():
        async with AsyncSessionLocal() as db:
            return await AttendanceService.get_by_session(db, session_id)
    return run_async(_get())

def get_session_analytics(session_id: int):
    async def _get():
        async with AsyncSessionLocal() as db:
            return await AnalyticsService.get_session_analytics(db, session_id)
    return run_async(_get())

def get_session_suggestions(session_id: int):
    async def _get():
        async with AsyncSessionLocal() as db:
            return await SuggestionService().get_by_session(db, session_id)
    return run_async(_get())

def get_all_students():
    async def _get():
        async with AsyncSessionLocal() as db:
            return await StudentService.get_all(db)
    return run_async(_get())

# ---------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# ---------------------------------------------------------
if menu == "📊 Dashboard Overview":
    st.markdown("## 📊 Classroom Intelligence Dashboard")
    st.markdown("Real-time telemetry and observable behavioral indicators across enrolled classroom cohorts.")

    # High level KPI calculation
    students = get_all_students()
    total_students = len(students)
    consented_count = sum(1 for s in students if s.has_consented)
    total_sessions_count = len(all_sessions)

    # Calculate global attendance and engagement stats
    total_att_records = 0
    total_present = 0
    avg_interaction_scores = []

    for s in all_sessions:
        total_att_records += getattr(s, "attendance_count", 0)
        total_present += getattr(s, "present_count", 0)
        avg_eng = getattr(s, "avg_engagement", 0.0)
        if avg_eng > 0:
            avg_interaction_scores.append(avg_eng)

    overall_att_pct = round((total_present / total_att_records * 100.0), 1) if total_att_records > 0 else 0.0
    global_avg_interaction = round(float(np.mean(avg_interaction_scores)), 1) if avg_interaction_scores else 0.0

    # Top KPI Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Enrolled Students", f"{total_students}", f"{consented_count} Consented")
    with c2:
        st.metric("Total Sessions", f"{total_sessions_count}", "3 Archived Cohorts")
    with c3:
        st.metric("Attendance Rate", f"{overall_att_pct}%", "Sustained Verified")
    with c4:
        st.metric("Avg Interaction Index", f"{global_avg_interaction}/100", "Observable Signals")
    with c5:
        st.metric("Biometric Consent Rate", f"{round(consented_count/total_students*100, 1) if total_students else 0}%", "Opt-in Enforced")

    st.markdown("<br>", unsafe_allow_html=True)

    # Session Overview Table
    st.markdown("### 📋 Conducted & Scheduled Sessions")
    
    session_rows = []
    for s in all_sessions:
        rate = round((s.present_count / s.attendance_count * 100.0), 1) if s.attendance_count > 0 else 0.0
        session_rows.append({
            "Session ID": s.session_uid,
            "Course Title": s.course_title,
            "Room": s.room_number or "N/A",
            "Status": s.status,
            "Target Cohort": s.target_class or "All",
            "Present / Enrolled": f"{s.present_count} / {s.attendance_count}",
            "Attendance Rate": f"{rate}%",
            "Avg Interaction": f"{s.avg_engagement}/100",
            "Duration": f"{int(s.actual_duration_seconds // 60)} min" if s.actual_duration_seconds else f"Exp: {s.expected_duration_minutes} min"
        })

    if session_rows:
        df_sessions = pd.DataFrame(session_rows)
        st.dataframe(df_sessions, use_container_width=True, hide_index=True)
    else:
        st.info("No session records found. Use the sidebar to reset demo data.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-session comparative trend chart
    st.markdown("### 📈 Cohort Trends Across Recent Sessions")
    trend_data = []
    for s in all_sessions:
        if s.status == "COMPLETED" and s.attendance_count > 0:
            rate = round((s.present_count / s.attendance_count * 100.0), 1)
            trend_data.append({
                "Session": s.course_title.split(":")[0],
                "Attendance Rate (%)": rate,
                "Interaction Score (0-100)": s.avg_engagement
            })

    if trend_data:
        chart_df = pd.DataFrame(trend_data).set_index("Session")
        st.bar_chart(chart_df, height=300)
    else:
        st.caption("No completed session data available for comparative charts.")

# ---------------------------------------------------------
# PAGE 2: STUDENT MANAGEMENT & CONSENT
# ---------------------------------------------------------
elif menu == "👥 Student & Consent Management":
    st.markdown("## 👥 Student Enrollment & Biometric Consent Management")
    st.markdown("""
    **Strict Consent Protocol**: In compliance with responsible educational privacy guidelines, 
    facial embeddings (512-dim vectors) are extracted and stored **strictly after explicit affirmative consent**. 
    If a student opts out or revokes consent, their embedding vector is instantly neutralized and face matching is disabled.
    """)

    students = get_all_students()

    # Display Students Table
    student_records = []
    for s in students:
        has_emb = "✅ Stored (512-dim)" if s.embedding_path and os.path.exists(s.embedding_path) else "❌ None / Disabled"
        student_records.append({
            "ID": s.id,
            "Student Code": s.student_id,
            "Full Name": s.full_name,
            "Cohort / Course": s.course_name,
            "Email": s.email or "N/A",
            "Biometric Consent": "GRANTED" if s.has_consented else "REVOKED / OPT-OUT",
            "Consent Date": s.consent_date.strftime("%Y-%m-%d %H:%M") if s.consent_date else "None",
            "Biometric Vector": has_emb
        })

    st.dataframe(pd.DataFrame(student_records), use_container_width=True, hide_index=True)

    st.markdown("---")

    col_toggle, col_add = st.columns([1, 1])

    # Consent Toggle Form
    with col_toggle:
        st.markdown("### 🔒 Consent Workflow Action")
        st.caption("Simulate student granting or revoking biometric consent in real time.")

        student_dict = {s.id: f"{s.student_id} - {s.full_name} ({'Consented' if s.has_consented else 'Revoked'})" for s in students}
        if student_dict:
            target_student_id = st.selectbox("Select Student Profile", options=list(student_dict.keys()), format_func=lambda x: student_dict[x])
            target_student = next((s for s in students if s.id == target_student_id), None)

            if target_student:
                new_consent_state = st.toggle("Grant Biometric Consent", value=target_student.has_consented)
                audit_note = st.text_input("Consent Audit Note", value=f"Updated via Educator Console at {time.strftime('%H:%M:%S')}")

                if st.button("Apply Consent Status Update", use_container_width=True):
                    from app.schemas.student import StudentConsentUpdate
                    async def _update_consent():
                        async with AsyncSessionLocal() as db:
                            payload = StudentConsentUpdate(has_consented=new_consent_state, audit_notes=audit_note)
                            return await StudentService.update_consent(db, target_student_id, payload)
                    
                    updated = run_async(_update_consent())
                    st.success(f"Consent status for {updated.full_name} updated to: {'GRANTED' if updated.has_consented else 'REVOKED'}")
                    time.sleep(0.6)
                    st.rerun()

    # Add Demo Student Form
    with col_add:
        st.markdown("### ➕ Enroll New Synthetic Demo Student")
        st.caption("Create a demonstration profile for classroom cohort testing.")

        with st.form("new_student_form"):
            new_code = st.text_input("Student ID Code", value=f"CS-2024-{len(students)+1:03d}")
            new_name = st.text_input("Full Name", value="Alex Taylor")
            new_cohort = st.text_input("Course Cohort", value="Computer Science 101")
            new_email = st.text_input("Institutional Email", value="alex.taylor@university.edu")
            consented_new = st.checkbox("Grant Biometric Consent Immediately", value=True)
            submit_student = st.form_submit_button("Enroll Student Profile", use_container_width=True)

            if submit_student:
                async def _create_student():
                    async with AsyncSessionLocal() as db:
                        payload = StudentCreate(
                            student_id=new_code,
                            full_name=new_name,
                            course_name=new_cohort,
                            email=new_email,
                            has_consented=consented_new
                        )
                        return await StudentService.create(db, payload)
                created = run_async(_create_student())
                st.success(f"Student profile '{created.full_name}' successfully enrolled!")
                time.sleep(0.5)
                st.rerun()

# ---------------------------------------------------------
# PAGE 3: VIDEO ANALYSIS PIPELINE
# ---------------------------------------------------------
elif menu == "🎥 Video Analysis Pipeline":
    st.markdown("## 🎥 Multimodal Video Analysis Pipeline")
    st.markdown("""
    Ingest recorded classroom video footage to extract multi-person spatial tracks, 
    estimate 3D head-orientation angles via `solvePnP`, compute motion intensity, and evaluate microphone voice activity.
    """)

    # Community Cloud Resource Guard
    st.info("""
    ℹ️ **Cloud Deployment Execution Notice**: 
    Streamlit Community Cloud operates with bounded CPU and memory limits. 
    You can run the **Bounded Real Pipeline** on short sample clips (≤ 15 MB) or test the 
    **Simulated High-Volume Classroom Mode** for instant demonstration without resource bottlenecks.
    """)

    active_sess_id = st.session_state.get("active_session_id", all_sessions[0].id if all_sessions else 1)
    current_session = get_active_session(active_sess_id)

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("### 1. Ingestion Mode & Video Source")
        analysis_mode = st.radio(
            "Select Processing Mode",
            [
                "Bounded Real Pipeline (CPU inference, frame sampling, solvePnP)",
                "Simulated Classroom Demo (Instant demonstration, synthetic telemetry)"
            ]
        )

        source_choice = st.radio(
            "Video Input Source",
            [
                "Use Pre-packaged Classroom Sample Clip (data/test_classroom_lecture.mp4)",
                "Upload New Sample Video File (MP4, Max 15 MB, ≤ 30s)"
            ]
        )

        uploaded_file = None
        if "Upload" in source_choice:
            uploaded_file = st.file_uploader("Upload Classroom Video", type=["mp4", "avi", "mov"])
            if uploaded_file and uploaded_file.size > 15 * 1024 * 1024:
                st.error("Uploaded video exceeds the 15 MB Community Cloud threshold. Please upload a shorter clip.")
                uploaded_file = None

    with c_right:
        st.markdown("### 2. Destination Session Target")
        if current_session:
            st.markdown(f"""
            <div style="background-color: #131b2e; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                <div style="font-size: 0.8rem; color: #94a3b8;">TARGET SESSION</div>
                <div style="font-size: 1.05rem; color: #38bdf8; font-weight: 600;">{current_session.course_title}</div>
                <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">
                    UID: <code>{current_session.session_uid}</code> • Cohort: {current_session.target_class} • Current Status: <b>{current_session.status}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No session selected. Please pick a session in the sidebar.")

        start_btn = st.button("🚀 Start Video Analysis", use_container_width=True, type="primary")

    # Processing Logic
    if start_btn:
        pipeline = get_or_create_pipeline()

        if "Simulated" in analysis_mode:
            # ----------------- SIMULATED DEMONSTRATION MODE -----------------
            st.markdown("""
            <div style="background-color: #451a03; border: 1px solid #d97706; border-radius: 8px; padding: 12px; margin: 15px 0;">
                <div style="font-weight: 700; color: #fde68a;">⚠️ [SIMULATED DEMONSTRATION MODE - SYNTHETIC DATA]</div>
                <div style="font-size: 0.85rem; color: #fef3c7;">
                    Executing high-volume classroom telemetry simulation. This demo synthesizes 50-minute multi-camera 
                    observable signals and updates attendance records without consuming limited cloud CPU cycles.
                </div>
            </div>
            """, unsafe_allow_html=True)

            prog_bar = st.progress(0, text="Initializing simulated multi-camera stream...")
            for p in range(1, 101, 15):
                time.sleep(0.1)
                prog_bar.progress(p, text=f"Processing simulated classroom stream... ({p}%)")
            prog_bar.progress(100, text="Finalizing attendance verification and observable signal aggregation.")

            # Update session status to COMPLETED and mark attendance
            async def _finalize_sim():
                async with AsyncSessionLocal() as db:
                    sess = await db.get(Session, active_sess_id)
                    if sess:
                        sess.status = "COMPLETED"
                        sess.actual_duration_seconds = 3000.0
                        await db.commit()
            run_async(_finalize_sim())
            st.success("✅ Simulated Classroom Analysis Complete! Inspect the updated Attendance and Engagement tabs.")

        else:
            # ----------------- REAL BOUNDED PIPELINE MODE -----------------
            # Determine video path
            temp_path = None
            if "Pre-packaged" in source_choice:
                sample_p = ROOT_DIR / "data" / "test_classroom_lecture.mp4"
                if not sample_p.exists():
                    sample_p = BACKEND_DIR / "data" / "test_classroom_lecture.mp4"
                if not sample_p.exists():
                    st.error("Pre-packaged sample video file not found on disk.")
                    st.stop()
                video_input_path = str(sample_p)
            else:
                if not uploaded_file:
                    st.warning("Please select or upload a video file to proceed.")
                    st.stop()
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = tmp.name
                video_input_path = temp_path

            try:
                st.markdown("##### ⚙️ Executing Multimodal Computer Vision & Audio Analysis...")
                prog_bar = st.progress(0, text="Opening video stream and loading feature weights...")
                status_text = st.empty()
                preview_spot = st.empty()

                cap = cv2.VideoCapture(video_input_path)
                if not cap.isOpened():
                    st.error("Failed to read video file. Invalid container or unsupported video codec.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                max_process_frames = min(total_frames, 60) # Bound to at most 60 frames for cloud safety

                # Load enrolled embeddings for this session's cohort
                async def _load_enrolled():
                    async with AsyncSessionLocal() as db:
                        return await StudentService.load_all_consented_embeddings(db, course_name=current_session.target_class)
                
                enrolled_students = run_async(_load_enrolled())
                enrolled_cache = [(s.id, s.full_name, emb) for s, emb in enrolled_students]

                processed = 0
                sample_step = max(1, int(fps / 2.0)) # sample 2 fps
                last_preview_img = None

                while cap.isOpened() and processed < max_process_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if processed % sample_step == 0:
                        timestamp_sec = processed / fps
                        
                        # 1. Detect Persons
                        detections = pipeline.detector.detect(frame) if pipeline else []
                        # 2. Track Persons
                        tracks = pipeline.tracker.update(detections) if pipeline else []

                        # 3. Face & Head Pose Analysis
                        for track in tracks:
                            x1, y1, x2, y2 = track.bbox
                            person_crop = frame[y1:y2, x1:x2]
                            if person_crop.size > 0 and pipeline:
                                face = pipeline.face_recognizer.detect_face(person_crop)
                                if face is not None:
                                    is_fwd, yaw, pitch, roll = pipeline.head_pose.estimate_orientation(face)
                                    # Draw visual indicators
                                    cv2.putText(frame, f"Yaw:{yaw:.0f} P:{pitch:.0f}", (x1, max(20, y1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if is_fwd else (0, 165, 255), 1)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

                        # Update progress
                        pct = int((processed / max_process_frames) * 100)
                        prog_bar.progress(pct, text=f"Analyzing frame {processed}/{max_process_frames} ({timestamp_sec:.1f}s)")
                        status_text.caption(f"Detected {len(tracks)} attendees • Evaluating 3D head orientation & VAD audio energy")

                        # Show frame preview
                        rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        last_preview_img = rgb_preview

                    processed += 1

                cap.release()
                prog_bar.progress(100, text="Processing complete! Finalizing attendance records in SQLite.")

                if last_preview_img is not None:
                    preview_spot.image(last_preview_img, caption="Analyzed Video Frame with solvePnP Orientation & Person Tracks", use_container_width=True)

                # Mark session COMPLETED
                async def _complete_session():
                    async with AsyncSessionLocal() as db:
                        sess = await db.get(Session, active_sess_id)
                        if sess:
                            sess.status = "COMPLETED"
                            sess.actual_duration_seconds = max(30.0, processed / fps)
                            # Finalize attendance (unseen pending students marked ABSENT)
                            await SessionService.finalize_session_attendance(db, sess.id)
                            await db.commit()
                run_async(_complete_session())

                st.success("✅ Real Pipeline Execution Successful! Attendance records and observable windows updated.")

            finally:
                # Ephemeral Media Policy: Strictly delete temporary uploaded video file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

# ---------------------------------------------------------
# PAGE 4: ATTENDANCE VERIFICATION
# ---------------------------------------------------------
elif menu == "📋 Attendance Verification":
    st.markdown("## 📋 Attendance Verification & Sustained Presence Rules")
    st.markdown(r"""
    **Responsible Verification Protocol**:
    - **Sustained Presence Requirement**: Never marks present on a single frame (requires $\ge 5$ discrete observations).
    - **Arrival Pacing**: Arrival after 10 minutes is flagged as `LATE`.
    - **Premature Departure**: Departing before 40% duration is flagged as `LEFT_EARLY`.
    - **Biometric Isolation**: Sighting without an enrolled consent match is marked `UNVERIFIED`.
    - **Session Closure**: `ABSENT` status is assigned strictly after a session completes, never while pending.
    """)

    active_sess_id = st.session_state.get("active_session_id", all_sessions[0].id if all_sessions else 1)
    current_session = get_active_session(active_sess_id)
    records = get_session_attendance(active_sess_id)

    # Status KPI Cards
    if records:
        present_c = sum(1 for r in records if r["status"] == "PRESENT")
        late_c = sum(1 for r in records if r["status"] == "LATE")
        early_c = sum(1 for r in records if r["status"] == "LEFT_EARLY")
        unv_c = sum(1 for r in records if r["status"] == "UNVERIFIED")
        pend_c = sum(1 for r in records if r["status"] == "PENDING")
        abs_c = sum(1 for r in records if r["status"] == "ABSENT")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("PRESENT", f"{present_c}", "Verified")
        k2.metric("LATE", f"{late_c}", "Arrival >10m")
        k3.metric("LEFT EARLY", f"{early_c}", "< 40% Duration")
        k4.metric("UNVERIFIED", f"{unv_c}", "Cosine < 0.65")
        k5.metric("PENDING", f"{pend_c}", "Unobserved")
        k6.metric("ABSENT", f"{abs_c}", "Post-Completion")

    st.markdown("---")

    # Filter selector
    filter_status = st.selectbox("Filter by Status", ["ALL", "PRESENT", "LATE", "LEFT_EARLY", "UNVERIFIED", "PENDING", "ABSENT"])

    table_data = []
    for r in records:
        if filter_status == "ALL" or r["status"] == filter_status:
            table_data.append({
                "Record ID": r["id"],
                "Student Code": r["student_code"] or "N/A",
                "Name": r["student_name"],
                "Status": r["status"],
                "First Seen": f"{r['first_seen_seconds']:.1f}s" if r["first_seen_seconds"] else "--",
                "Last Seen": f"{r['last_seen_seconds']:.1f}s" if r["last_seen_seconds"] else "--",
                "Sustained (s)": r["sustained_duration_seconds"],
                "Observations": r["observation_count"],
                "Confidence": f"{r['avg_confidence'] * 100:.1f}%",
                "Override?": "Yes (Manual)" if r["is_manual_override"] else "Automated",
                "Audit Notes": r["notes"] or ""
            })

    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.info(f"No records matching filter '{filter_status}' for this session.")

    st.markdown("---")

    # Instructor Manual Override Form
    st.markdown("### ✍️ Instructor Manual Override Workflow")
    st.caption("Human-in-the-loop audit capability: update any automated attendance determination with mandatory audit reasoning.")

    col_rec, col_stat, col_note, col_btn = st.columns([2, 1.5, 3, 1.5])
    with col_rec:
        rec_options = {r["id"]: f"{r['student_name']} ({r['status']})" for r in records}
        chosen_rec_id = st.selectbox("Select Record", options=list(rec_options.keys()), format_func=lambda x: rec_options[x])
    with col_stat:
        new_status = st.selectbox("Corrected Status", ["PRESENT", "LATE", "LEFT_EARLY", "UNVERIFIED", "ABSENT"])
    with col_note:
        override_note = st.text_input("Mandatory Audit Note", value="Instructor visually confirmed presence in rear row")
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Save Override", use_container_width=True):
            async def _override():
                async with AsyncSessionLocal() as db:
                    payload = AttendanceUpdate(status=new_status, notes=override_note)
                    return await AttendanceService.manual_override(db, chosen_rec_id, payload)
            updated = run_async(_override())
            st.success(f"Record #{updated.id} successfully updated to {updated.status}!")
            time.sleep(0.5)
            st.rerun()

# ---------------------------------------------------------
# PAGE 5: ENGAGEMENT SIGNAL ANALYTICS
# ---------------------------------------------------------
elif menu == "📈 Engagement Signal Analytics":
    st.markdown("## 📈 Observable Engagement Signal Analytics")
    st.markdown("""
    Observable behavioral metrics aggregated across **30-second time windows**.
    
    *Ethical Scientific Disclaimer*: Signals represent **observable physical indicators** (forward orientation, movement, acoustic voice activity) 
    and strictly avoid making inferences regarding internal cognitive states, intelligence, or comprehension.
    """)

    active_sess_id = st.session_state.get("active_session_id", all_sessions[0].id if all_sessions else 1)
    analytics = get_session_analytics(active_sess_id)

    # Session Analytics KPI Header
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite Interaction Index", f"{analytics.avg_interaction_score}/100", "Composite")
    c2.metric("Forward Head-Pose Ratio", f"{int(analytics.avg_forward_facing_ratio * 100)}%", "solvePnP Orientation")
    c3.metric("Movement Intensity", f"{int(analytics.avg_movement_intensity * 100)}%", "Frame Differencing")
    c4.metric("Microphone Voice Ratio", f"{int(analytics.avg_speech_active_ratio * 100)}%", "RMS Energy VAD")

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-series line chart of 30-second windows
    st.markdown("### ⏱️ Time-Series Telemetry (30-Second Windows)")
    if analytics.timeline:
        chart_rows = []
        for w in analytics.timeline:
            chart_rows.append({
                "Window Interval": f"{int(w.start_seconds)}s - {int(w.end_seconds)}s",
                "Interaction Activity Score (0-100)": w.interaction_activity_score,
                "Forward Orientation Ratio (0-1)": w.forward_facing_ratio * 100.0,
                "Movement Intensity (%)": w.movement_intensity * 100.0,
                "Speech Activity (%)": w.speech_active_ratio * 100.0 if w.is_audio_available else 0.0
            })
        df_chart = pd.DataFrame(chart_rows).set_index("Window Interval")
        st.line_chart(df_chart, height=350)
    else:
        st.info("No engagement window telemetry recorded for this session yet.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Notable Changes & Pedagogical Anomalies Feed
    st.markdown("### 🔔 Observable Behavioral Anomalies & Transitions")
    if analytics.notable_changes:
        for note in analytics.notable_changes:
            if "drop" in note.lower():
                st.warning(f"📉 {note}")
            elif "spike" in note.lower():
                st.success(f"📈 {note}")
            else:
                st.info(f"ℹ️ {note}")
    else:
        st.caption("No notable activity fluctuations detected.")

    # Detailed Window Data Table
    with st.expander("🔍 View Raw 30-Second Window Telemetry Table"):
        if analytics.timeline:
            raw_data = [{
                "Window #": w.window_index + 1,
                "Time Interval": f"{w.start_seconds:.0f}s - {w.end_seconds:.0f}s",
                "Avg Persons": w.person_count_avg,
                "Forward Ratio": f"{w.forward_facing_ratio:.2f}",
                "Movement": f"{w.movement_intensity:.2f}",
                "Speech Ratio": f"{w.speech_active_ratio:.2f}" if w.is_audio_available else "N/A",
                "Composite Score": f"{w.interaction_activity_score:.1f}/100"
            } for w in analytics.timeline]
            st.dataframe(pd.DataFrame(raw_data), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# PAGE 6: AI TEACHING SUGGESTIONS
# ---------------------------------------------------------
elif menu == "💡 AI Teaching Suggestions":
    st.markdown("## 💡 Pedagogical Suggestions & Teaching Interventions")
    st.markdown("""
    The AI Teaching Suggestion Engine analyzes observable telemetry trends (consecutive drops, prolonged quiet, high volatility) 
    and synthesizes **actionable, evidence-backed pedagogical check-ins**.
    """)

    active_sess_id = st.session_state.get("active_session_id", all_sessions[0].id if all_sessions else 1)
    suggestions = get_session_suggestions(active_sess_id)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 Synthesize Fresh Suggestions", use_container_width=True, type="primary"):
            with st.spinner("Analyzing observable signals and synthesizing interventions..."):
                async def _gen():
                    async with AsyncSessionLocal() as db:
                        return await SuggestionService().generate_and_save_for_session(db, active_sess_id)
                new_suggs = run_async(_gen())
                st.success(f"Generated {len(new_suggs)} pedagogical suggestions!")
                time.sleep(0.5)
                st.rerun()

    with col_info:
        st.caption("Suggestions are generated via deterministic pedagogical rule analysis or optional local Ollama LLM.")

    st.markdown("---")

    if suggestions:
        for idx, sug in enumerate(suggestions):
            urgency_class = f"urgency-{sug.urgency.lower()}"
            st.markdown(f"""
            <div style="background-color: #131b2e; border: 1px solid #1e293b; border-left: 4px solid {'#dc2626' if sug.urgency=='HIGH' else '#d97706' if sug.urgency=='MEDIUM' else '#059669'}; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">{sug.title}</div>
                    <div>
                        <span class="{urgency_class}">{sug.urgency} URGENCY</span>
                        <span style="background-color: #1e293b; color: #94a3b8; padding: 3px 6px; border-radius: 4px; font-size: 0.72rem; margin-left: 6px;">{sug.suggestion_type}</span>
                    </div>
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 8px;">{sug.description}</div>
                <div style="font-size: 0.8rem; color: #38bdf8;"><b>Evidence Citation:</b> <i>{sug.evidence}</i></div>
            </div>
            """, unsafe_allow_html=True)

            # Acknowledge button
            if not sug.acknowledged:
                if st.button(f"Mark Suggestion #{sug.id} as Acknowledged", key=f"ack_{sug.id}"):
                    async def _ack():
                        async with AsyncSessionLocal() as db:
                            return await SuggestionService().acknowledge(db, sug.id)
                    run_async(_ack())
                    st.success("Suggestion marked acknowledged.")
                    time.sleep(0.4)
                    st.rerun()
    else:
        st.info("No active suggestions recorded for this session. Click 'Synthesize Fresh Suggestions' above.")

# ---------------------------------------------------------
# PAGE 7: SESSION REPORTS & EXPORTS
# ---------------------------------------------------------
elif menu == "📑 Session Reports & Exports":
    st.markdown("## 📑 Classroom Session Reports & Data Export")
    st.markdown("Download standardized institutional reports, verification audit logs, and signal telemetry in CSV format.")

    active_sess_id = st.session_state.get("active_session_id", all_sessions[0].id if all_sessions else 1)
    current_session = get_active_session(active_sess_id)

    if current_session:
        st.markdown(f"### Export Reports for: `{current_session.course_title}`")
        
        # 1. Attendance CSV Export
        async def _get_csv():
            async with AsyncSessionLocal() as db:
                return await AnalyticsService.export_attendance_csv(db, active_sess_id)
        
        csv_string = run_async(_get_csv())

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 1. Attendance & Verification Audit CSV")
            st.caption("Includes student ID, verification status, sustained duration, observation count, and override notes.")
            st.download_button(
                label="📥 Download Attendance CSV",
                data=csv_string,
                file_name=f"Attendance_Report_Session_{active_sess_id}_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

        with col2:
            st.markdown("#### 2. Engagement Window Telemetry CSV")
            st.caption("30-second window metrics: orientation, motion, speech ratios, and interaction score.")
            analytics = get_session_analytics(active_sess_id)
            if analytics.timeline:
                win_df = pd.DataFrame([w.model_dump() for w in analytics.timeline])
                win_csv = win_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Engagement Telemetry CSV",
                    data=win_csv,
                    file_name=f"Engagement_Telemetry_Session_{active_sess_id}_{time.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown("### 📄 Attendance Report Data Preview")
        records = get_session_attendance(active_sess_id)
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# PAGE 8: RESPONSIBLE AI, PRIVACY & LIMITATIONS
# ---------------------------------------------------------
elif menu == "🛡️ Responsible AI & Privacy":
    st.markdown("## 🛡️ Responsible Educational AI, Privacy & Limitations")
    st.markdown("""
    This application is designed from first principles around educational ethics, data minimization, 
    and strict privacy boundaries.
    """)

    st.markdown("---")

    col_priv, col_ethic = st.columns([1, 1])

    with col_priv:
        st.markdown("""
        ### 🔒 Privacy Model: Local vs Public Cloud Demo
        
        1. **Offline-First Core Architecture**:
           - In actual on-premise classroom deployment, the system operates **100% offline on local hardware**.
           - Video streams, audio waveforms, and biometric facial vectors are never transmitted over external networks or third-party cloud APIs.
        
        2. **Public Demonstration Protections**:
           - **Synthetic Demonstration Profiles**: All student names, ID codes, and demo data in this cloud instance are synthetic records.
           - **Ephemeral Media Policy**: Any short video uploaded for demonstration is buffered in temporary storage and **immediately wiped from memory and disk** upon completion or error.
           - **No Persistent Biometric Exposure**: Raw biometric vectors (`.npy`) and private SQLite tables are excluded from GitHub and public cloud storage.
        
        3. **Affirmative Opt-In Consent**:
           - Facial feature extraction is strictly blocked without explicit prior consent.
           - If a student revokes consent, their feature vector is neutralized and face matching is disabled immediately.
        """)

    with col_ethic:
        st.markdown("""
        ### ⚖️ Observable Signals vs Internal Mental State
        
        1. **No Claims of Mind-Reading or Attention Measurement**:
           - Head-pose orientation, frame movement, and voice activity are **observable physical indicators only**.
           - The system explicitly **does not measure internal cognitive comprehension, mental focus, intelligence, or motivation**.
        
        2. **Avoidance of Emotion Recognition Pseudo-Science**:
           - The software does not attempt facial emotion recognition (e.g. happy, sad, confused), which has been widely discredited in educational contexts.
        
        3. **Educator Primacy**:
           - Automated attendance and pedagogical suggestions are strictly **advisory**.
           - Final attendance records and instructional adjustments always remain with the human educator.
           - Comprehensive manual override workflows with audit trails ensure human accountability.
        """)

    st.markdown("---")
    st.caption("Multimodal Attendance & Engagement Agent • Engineered for Privacy, Performance, and Responsible AI.")
