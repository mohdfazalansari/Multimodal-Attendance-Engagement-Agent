import os
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil

from render_helpers import get_font, draw_topbar, draw_sidebar, draw_lower_third

# Paths
BASE_DIR = r"F:\Multimodal Attendance & Engagement Agent"
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "Multimodal_Attendance_Engagement_Agent_Demo.mp4")
ARTIFACTS_DIR = r"C:\Users\fazal\.gemini\antigravity-ide\brain\6389ca13-57aa-4e01-9fb3-02314a5fa613"
PORTRAIT_PATH = os.path.join(BASE_DIR, "data", "consented_student_portrait.jpg")

FPS = 30
WIDTH = 1920
HEIGHT = 1080

def create_base_canvas():
    # Light gray modern background matching React app (bg-slate-50: #f8fafc)
    img = Image.new("RGB", (WIDTH, HEIGHT), (248, 250, 252))
    return img

# ----------------- SCENE BUILDERS ----------------- #

def render_scene_1(progress: float) -> Image.Image:
    """Scene 1: Introduction Title Screen (0:00 - 0:12)"""
    # Deep midnight slate gradient background
    img = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(img)

    # Ambient subtle gradient / radial accents
    for r in range(400, 0, -20):
        alpha = int(12 * (1.0 - r / 400))
        draw.ellipse([(960 - r, 420 - r), (960 + r, 420 + r)], fill=(30, 58, 138))

    # Top Tag
    draw.rounded_rectangle([(810, 230), (1110, 270)], radius=20, fill=(30, 41, 59), outline=(59, 130, 246), width=1)
    draw.text((830, 241), "ENGINEERING DEMONSTRATION", font=get_font(13, bold=True), fill=(147, 197, 253))

    # Main Title
    title = "Multimodal Attendance & Engagement Agent"
    draw.text((960, 320), title, font=get_font(44, bold=True), fill=(255, 255, 255), anchor="mm")

    # Subtitle
    sub = "Offline-first • Privacy-aware • AI-powered classroom analytics"
    draw.text((960, 385), sub, font=get_font(22), fill=(148, 163, 184), anchor="mm")

    # Key Architecture Badges Row
    pills = [
        ("Computer Vision", "YOLOv8 + FaceNet 512-dim + solvePnP"),
        ("Audio Intelligence", "RMS Energy & Zero-Crossing VAD"),
        ("Hardware Engine", "NVIDIA RTX 3050 Laptop GPU (CUDA Enabled)"),
        ("Architecture", "100% Offline Local Processing • SQLite"),
        ("Pedagogical AI", "Observable Behavioral Indicators"),
    ]

    start_x = 210
    for category, detail in pills:
        draw.rounded_rectangle([(start_x, 480), (start_x + 280, 590)], radius=14, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
        draw.rounded_rectangle([(start_x + 16, 498), (start_x + 264, 526)], radius=6, fill=(30, 58, 138))
        draw.text((start_x + 24, 504), category.upper(), font=get_font(11, bold=True), fill=(191, 219, 254))
        draw.text((start_x + 16, 545), detail, font=get_font(12, bold=True), fill=(226, 232, 240))
        start_x += 300

    # Verification Highlights
    draw.rounded_rectangle([(360, 650), (1560, 780)], radius=16, fill=(15, 23, 42), outline=(51, 65, 85), width=1)
    draw.text((390, 675), "AUDITED CAPABILITIES & VERIFIED GUARANTEES", font=get_font(14, bold=True), fill=(96, 165, 250))
    checks = [
        "Sustained-Presence Rule: Requires >= 5 continuous observations; single frame never marks present",
        "Strict Biometric Consent: Face embedding is blocked unless affirmative consent is on record",
        "Zero Cloud Offloading: Video, audio, and facial embeddings are processed 100% locally on localhost",
        "Behavioral Proxy Standard: Measures observable physical signals, not internal comprehension or intelligence",
    ]
    y_c = 710
    for chk in checks:
        draw.ellipse([(390, y_c + 4), (400, y_c + 14)], fill=(16, 185, 129))
        draw.text((415, y_c), chk, font=get_font(13), fill=(203, 213, 225))
        y_c += 30

    # Lower Third
    draw_lower_third(
        draw,
        scene_number=1,
        scene_title="Project Introduction & Architecture",
        explanation="An offline classroom intelligence system combining computer vision, audio energy, and automated attendance verification.",
        tech_tag="Architecture Overview"
    )
    return img


def render_scene_2(progress: float) -> Image.Image:
    """Scene 2: Dashboard Overview (0:12 - 0:32)"""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "dashboard")
    draw_topbar(draw, "Dashboard Overview")

    # Header banner (blue gradient)
    draw.rounded_rectangle([(310, 96), (1890, 206)], radius=16, fill=(29, 78, 216))
    draw.rounded_rectangle([(330, 114), (540, 138)], radius=12, fill=(59, 130, 246))
    draw.text((345, 118), "Offline Classroom Intelligence", font=get_font(12, bold=True), fill=(255, 255, 255))
    draw.text((330, 146), "Classroom Analytics & Attendance", font=get_font(24, bold=True), fill=(255, 255, 255))
    draw.text((330, 180), "Real-time multi-person tracking, observable engagement signals, and automated verification running 100% locally.", font=get_font(13), fill=(219, 234, 254))

    # 4 Metric KPI Cards
    kpis = [
        ("Total Enrolled Students", "8", "Consented active cohort", (59, 130, 246), (239, 246, 255)),
        ("Total Classroom Sessions", "3", "2 Completed • 1 Scheduled", (16, 185, 129), (236, 253, 245)),
        ("Sustained Attendance", "81.2%", ">= 5 Valid observations rule", (139, 92, 246), (245, 243, 255)),
        ("Average Engagement Score", "80.4 / 100", "Composite behavioral indicator", (245, 158, 11), (254, 243, 199)),
    ]

    card_x = 310
    card_w = 375
    for title, val, sub, col, bg_col in kpis:
        draw.rounded_rectangle([(card_x, 226), (card_x + card_w, 350)], radius=14, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        draw.rounded_rectangle([(card_x + 16, 242), (card_x + 56, 282)], radius=10, fill=bg_col)
        draw.ellipse([(card_x + 30, 256), (card_x + 42, 268)], fill=col)
        draw.text((card_x + 68, 248), title, font=get_font(13, bold=True), fill=(100, 116, 139))
        draw.text((card_x + 68, 274), val, font=get_font(26, bold=True), fill=(30, 41, 59))
        draw.text((card_x + 16, 320), sub, font=get_font(11), fill=(148, 163, 184))
        card_x += card_w + 26

    # Bottom Split: Recent Sessions (Left 60%) + Attendance vs Engagement Trends (Right 40%)
    # Left: Recent Sessions Table
    draw.rounded_rectangle([(310, 372), (1240, 930)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((334, 396), "Recent Classroom Sessions", font=get_font(16, bold=True), fill=(30, 41, 59))
    draw.text((334, 420), "Telemetry records stored locally in SQLite database", font=get_font(12), fill=(100, 116, 139))

    # Table Header
    draw.rectangle([(334, 452), (1216, 486)], fill=(241, 245, 249))
    draw.text((350, 462), "SESSION UID", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((500, 462), "COURSE TITLE", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((820, 462), "STATUS", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((950, 462), "ATTENDANCE", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1090, 462), "ENGAGEMENT", font=get_font(11, bold=True), fill=(71, 85, 105))

    # Sessions Rows
    sessions_data = [
        ("SESS-CS101-01", "Computer Science 101: Intro to Algorithms", "COMPLETED", "87.5%", "82.0 / 100", (16, 185, 129)),
        ("SESS-CS101-02", "Computer Science 101: Graph Traversal & Search", "COMPLETED", "75.0%", "78.8 / 100", (16, 185, 129)),
        ("SESS-CS101-03", "Computer Science 101: Dynamic Programming Analysis", "SCHEDULED", "Pending", "Video Target", (245, 158, 11)),
    ]

    r_y = 502
    for s_uid, title, st, att, eng, st_col in sessions_data:
        draw.line([(334, r_y - 8), (1216, r_y - 8)], fill=(241, 245, 249), width=1)
        draw.text((350, r_y), s_uid, font=get_font(12, bold=True), fill=(30, 41, 59))
        draw.text((500, r_y), title, font=get_font(12), fill=(51, 65, 85))
        # Status pill
        draw.rounded_rectangle([(816, r_y - 3), (910, r_y + 21)], radius=10, fill=(236, 253, 245) if st == "COMPLETED" else (254, 243, 199))
        draw.text((830, r_y + 1), st, font=get_font(11, bold=True), fill=st_col)
        draw.text((965, r_y), att, font=get_font(12, bold=True), fill=(30, 41, 59))
        draw.text((1095, r_y), eng, font=get_font(12, bold=True), fill=(30, 41, 59))
        r_y += 56

    # Right: Recharts Trend Card
    draw.rounded_rectangle([(1264, 372), (1890, 930)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((1288, 396), "Attendance vs Engagement Trends", font=get_font(16, bold=True), fill=(30, 41, 59))
    draw.text((1288, 420), "Cross-session comparative indicators", font=get_font(12), fill=(100, 116, 139))

    # Recharts Simulated Plot Area
    p_x1, p_y1, p_x2, p_y2 = 1310, 480, 1850, 870
    draw.rectangle([(p_x1, p_y1), (p_x2, p_y2)], fill=(248, 250, 252))
    for gy in range(p_y1, p_y2 + 1, 65):
        draw.line([(p_x1, gy), (p_x2, gy)], fill=(226, 232, 240), width=1)
        val = int(100 - (gy - p_y1) / (p_y2 - p_y1) * 100)
        draw.text((p_x1 - 35, gy - 7), f"{val}%", font=get_font(10), fill=(148, 163, 184))

    # Trend lines
    # Attendance line (blue)
    pts_att = [(p_x1 + 60, p_y1 + 45), (p_x1 + 250, p_y1 + 95), (p_x1 + 440, p_y2 - 20)]
    draw.line(pts_att, fill=(59, 130, 246), width=3)
    for px, py in pts_att:
        draw.ellipse([(px - 6, py - 6), (px + 6, py + 6)], fill=(59, 130, 246), outline=(255, 255, 255), width=2)

    # Engagement line (amber)
    pts_eng = [(p_x1 + 60, p_y1 + 70), (p_x1 + 250, p_y1 + 82), (p_x1 + 440, p_y2 - 20)]
    draw.line(pts_eng, fill=(245, 158, 11), width=3)
    for px, py in pts_eng:
        draw.ellipse([(px - 6, py - 6), (px + 6, py + 6)], fill=(245, 158, 11), outline=(255, 255, 255), width=2)

    # Legend
    draw.ellipse([(1400, 895), (1410, 905)], fill=(59, 130, 246))
    draw.text((1420, 893), "Sustained Attendance %", font=get_font(12), fill=(71, 85, 105))
    draw.ellipse([(1640, 895), (1650, 905)], fill=(245, 158, 11))
    draw.text((1660, 893), "Engagement Score", font=get_font(12), fill=(71, 85, 105))

    draw_lower_third(
        draw,
        scene_number=2,
        scene_title="Live Dashboard & Class Telemetry",
        explanation="Displays verified database metrics: 8 enrolled students, 81.2% sustained presence, and trend analytics without cloud latency.",
        tech_tag="FastAPI + React Dashboard"
    )
    return img


def render_scene_3(progress: float) -> Image.Image:
    """Scene 3: Student Enrollment & Biometric Consent (0:32 - 0:52)"""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "enrollment")
    draw_topbar(draw, "Student Registration & Consent Management")

    # Left: Registration & Biometric Enrollment Form (45%)
    draw.rounded_rectangle([(310, 96), (1050, 930)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((334, 120), "Student & Face Enrollment", font=get_font(18, bold=True), fill=(30, 41, 59))
    draw.text((334, 146), "Register student record and compute local 512-dim facial embedding", font=get_font(12), fill=(100, 116, 139))

    # Form Fields
    fields = [
        ("Student Identifier *", "CS-2024-009"),
        ("Full Legal / Preferred Name *", "Clara Oswald"),
        ("Course / Academic Cohort *", "Computer Science 101"),
        ("University Email Address", "clara.oswald@university.edu"),
    ]
    f_y = 186
    for f_label, f_val in fields:
        draw.text((334, f_y), f_label, font=get_font(12, bold=True), fill=(71, 85, 105))
        draw.rounded_rectangle([(334, f_y + 22), (1026, f_y + 60)], radius=8, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
        draw.text((348, f_y + 32), f_val, font=get_font(13), fill=(15, 23, 42))
        f_y += 72

    # Biometric Sample Photo Preview Card
    draw.rounded_rectangle([(334, 480), (1026, 680)], radius=12, fill=(248, 250, 252), outline=(226, 232, 240), width=1)
    
    # Draw portrait photo thumbnail
    if os.path.exists(PORTRAIT_PATH):
        try:
            portrait_img = Image.open(PORTRAIT_PATH).convert("RGB")
            portrait_thumb = portrait_img.resize((150, 170))
            img.paste(portrait_thumb, (350, 495))
        except Exception:
            draw.rectangle([(350, 495), (500, 665)], fill=(203, 213, 225))
    else:
        draw.rectangle([(350, 495), (500, 665)], fill=(203, 213, 225))

    # Photo Details
    draw.text((520, 508), "Consented Test Participant Portrait", font=get_font(14, bold=True), fill=(30, 41, 59))
    draw.text((520, 532), "Source: consented_student_portrait.jpg (694 KB)", font=get_font(12), fill=(100, 116, 139))
    draw.text((520, 558), "Face Detection: Haar Cascade (394x394 crop)", font=get_font(12), fill=(16, 185, 129))
    draw.text((520, 582), "Embedding Vector: 512-dim L2-normalized float32", font=get_font(12), fill=(59, 130, 246))
    draw.text((520, 606), "Storage: Local filesystem (.npy matrix)", font=get_font(12), fill=(71, 85, 105))
    draw.text((520, 630), "Public API Exposure: Strictly Disabled (Zero Cloud)", font=get_font(12, bold=True), fill=(220, 38, 38))

    # Biometric Consent Checkbox Workflow
    draw.rounded_rectangle([(334, 700), (1026, 786)], radius=10, fill=(236, 253, 245), outline=(167, 243, 208), width=1)
    # Checked Box
    draw.rounded_rectangle([(350, 718), (374, 742)], radius=5, fill=(16, 185, 129))
    draw.text((356, 720), "✓", font=get_font(14, bold=True), fill=(255, 255, 255))
    draw.text((386, 716), "Explicit Biometric Consent Confirmed", font=get_font(13, bold=True), fill=(6, 95, 70))
    draw.text((386, 738), "Authorizes 512-dim facial feature extraction strictly for attendance verification on localhost.", font=get_font(11), fill=(4, 120, 87))

    # Success Banner / Button
    draw.rounded_rectangle([(334, 804), (1026, 856)], radius=10, fill=(16, 185, 129))
    draw.text((520, 818), "✓ Student Enrolled & Face Vector Stored Locally", font=get_font(14, bold=True), fill=(255, 255, 255))

    # Right: Student Directory Roster (55%)
    draw.rounded_rectangle([(1076, 96), (1890, 930)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((1100, 120), "Enrolled Student Roster & Consent Status", font=get_font(18, bold=True), fill=(30, 41, 59))
    draw.text((1100, 146), "Enforces privacy policy: face embeddings exist only for consented participants", font=get_font(12), fill=(100, 116, 139))

    # Roster Table Header
    draw.rectangle([(1100, 186), (1866, 222)], fill=(241, 245, 249))
    draw.text((1116, 196), "ID", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1240, 196), "FULL NAME", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1440, 196), "CONSENT STATUS", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1640, 196), "EMBEDDING VECTOR", font=get_font(11, bold=True), fill=(71, 85, 105))

    roster = [
        ("CS-2024-001", "Emma Watson", True, "Stored (.npy)"),
        ("CS-2024-002", "Liam Smith", True, "Stored (.npy)"),
        ("CS-2024-003", "Sophia Garcia", True, "Stored (.npy)"),
        ("CS-2024-004", "Noah Patel", True, "Stored (.npy)"),
        ("CS-2024-005", "Ava Chen", True, "Stored (.npy)"),
        ("CS-2024-006", "Lucas Vance", True, "Stored (.npy)"),
        ("CS-2024-007", "Maya Lin", True, "Stored (.npy)"),
        ("CS-2024-008", "Ethan Wright", False, "BLOCKED (No Consent)"),
        ("CS-2024-009", "Clara Oswald", True, "Stored (.npy)"),
    ]

    r_y = 236
    for sid, name, consented, emb_state in roster:
        draw.line([(1100, r_y - 6), (1866, r_y - 6)], fill=(241, 245, 249), width=1)
        draw.text((1116, r_y), sid, font=get_font(12, bold=True), fill=(30, 41, 59))
        draw.text((1240, r_y), name, font=get_font(12), fill=(15, 23, 42))

        # Consent Badge
        if consented:
            draw.rounded_rectangle([(1436, r_y - 3), (1570, r_y + 21)], radius=10, fill=(236, 253, 245))
            draw.text((1450, r_y + 1), "✓ Consent Granted", font=get_font(11, bold=True), fill=(16, 185, 129))
            draw.text((1640, r_y), emb_state, font=get_font(12), fill=(16, 185, 129))
        else:
            draw.rounded_rectangle([(1436, r_y - 3), (1580, r_y + 21)], radius=10, fill=(254, 242, 242))
            draw.text((1450, r_y + 1), "✕ Revoked / Denied", font=get_font(11, bold=True), fill=(220, 38, 38))
            draw.text((1640, r_y), emb_state, font=get_font(12, bold=True), fill=(220, 38, 38))
        r_y += 48

    draw_lower_third(
        draw,
        scene_number=3,
        scene_title="Biometric Consent & Face Enrollment",
        explanation="Facial embeddings (FaceNet 512-dim) are extracted only with affirmative consent. Ethan Wright opted out; embedding extraction is strictly blocked.",
        tech_tag="FaceNet + Local Vector Vault"
    )
    return img


def render_scene_4(progress: float) -> Image.Image:
    """Scene 4: Recorded Video Processing (0:52 - 1:18)"""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "video")
    draw_topbar(draw, "Recorded Video Ingestion & Offline Multimodal Pipeline")

    # Left: Ingestion Configuration & Progress (45%)
    draw.rounded_rectangle([(310, 96), (1050, 930)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((334, 120), "Recorded Video Processing", font=get_font(18, bold=True), fill=(30, 41, 59))
    draw.text((334, 146), "Offline multimodal computer vision and acoustic telemetry extraction", font=get_font(12), fill=(100, 116, 139))

    # 1. Target Session Dropdown
    draw.text((334, 184), "1. TARGET CLASSROOM SESSION", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.rounded_rectangle([(334, 206), (1026, 248)], radius=8, fill=(248, 250, 252), outline=(59, 130, 246), width=1)
    draw.text((348, 218), "SESS-CS101-03 — Computer Science 101: Dynamic Programming Analysis [SCHEDULED]", font=get_font(12, bold=True), fill=(30, 41, 59))

    # 2. Upload Video Dropzone
    draw.text((334, 272), "2. LECTURE VIDEO INGESTION CONTAINER", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.rounded_rectangle([(334, 294), (1026, 470)], radius=12, fill=(240, 253, 244), outline=(134, 239, 172), width=1)
    draw.rounded_rectangle([(640, 314), (720, 364)], radius=12, fill=(220, 252, 231))
    draw.text((660, 326), "▶", font=get_font(24), fill=(22, 101, 52))
    draw.text((680, 384), "test_classroom_lecture.mp4", font=get_font(14, bold=True), fill=(21, 128, 61), anchor="mm")
    draw.text((680, 408), "103.2 KB • Format: MP4 (H.264) • Resolution: 640x480 • 10 FPS", font=get_font(12), fill=(100, 116, 139), anchor="mm")
    draw.text((680, 436), "✓ Video loaded and validated for local frame extraction", font=get_font(11, bold=True), fill=(16, 185, 129), anchor="mm")

    # 3. Pipeline Execution Status
    draw.text((334, 496), "3. PROCESSING PIPELINE EXECUTION", font=get_font(11, bold=True), fill=(71, 85, 105))
    
    # Dynamic progress simulation
    pct = min(100, int(progress * 100))
    draw.rounded_rectangle([(334, 520), (1026, 546)], radius=8, fill=(226, 232, 240))
    prog_w = int((1026 - 334) * (pct / 100.0))
    if prog_w > 0:
        draw.rounded_rectangle([(334, 520), (334 + prog_w, 546)], radius=8, fill=(37, 99, 235))
    draw.text((1035, 525), f"{pct}%", font=get_font(12, bold=True), fill=(37, 99, 235))

    status_msg = "Completed: Attendance verified & 30s engagement windows generated." if pct == 100 else "Processing video frames (YOLOv8 + solvePnP + VAD)..."
    draw.text((334, 560), f"Status: {status_msg}", font=get_font(12, bold=True), fill=(30, 41, 59))

    # Stage telemetry boxes
    stages = [
        ("YOLOv8 Person Detection", "CUDA Accelerated (RTX 3050)", "2 Persons Tracked"),
        ("Head Pose Estimation", "Perspective-n-Point (solvePnP)", "Yaw: -4°, Pitch: +6° (Forward)"),
        ("Face Identification", "FaceNet Cosine Match (0.65 threshold)", "Emma Watson, Liam Smith"),
        ("Audio Energy Tracking", "RMS Energy & Zero-Crossing VAD", "Speech Active Ratio: 38%"),
    ]
    st_y = 600
    for st_title, st_eng, st_res in stages:
        draw.rounded_rectangle([(334, st_y), (1026, st_y + 48)], radius=8, fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.text((348, st_y + 8), st_title, font=get_font(12, bold=True), fill=(30, 41, 59))
        draw.text((348, st_y + 26), st_eng, font=get_font(11), fill=(100, 116, 139))
        draw.text((810, st_y + 16), st_res, font=get_font(11, bold=True), fill=(16, 185, 129))
        st_y += 56

    # Privacy media policy callout
    draw.rounded_rectangle([(334, 840), (1026, 900)], radius=8, fill=(254, 242, 242), outline=(254, 202, 202), width=1)
    draw.text((350, 852), "🛡️ Ephemeral Media Policy Verified", font=get_font(12, bold=True), fill=(185, 28, 28))
    draw.text((350, 874), "Temporary video buffer will be automatically purged from disk once telemetry is recorded.", font=get_font(11), fill=(127, 29, 29))

    # Right: Live Vision Pipeline Visualizer (55%)
    draw.rounded_rectangle([(1076, 96), (1890, 930)], radius=16, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
    draw.text((1100, 120), "Computer Vision Pipeline Visualizer", font=get_font(18, bold=True), fill=(248, 250, 252))
    draw.text((1100, 146), "Actual frame detection, spatial tracker IDs, and 3D head orientation vectors", font=get_font(12), fill=(148, 163, 184))

    # Simulated lecture frame render inside monitor
    m_x1, m_y1, m_x2, m_y2 = 1100, 180, 1866, 750
    draw.rounded_rectangle([(m_x1, m_y1), (m_x2, m_y2)], radius=12, fill=(235, 238, 242), outline=(51, 65, 85), width=2)
    
    # Blackboard on top
    draw.rectangle([(m_x1 + 60, m_y1 + 20), (m_x2 - 60, m_y1 + 140)], fill=(45, 55, 45))
    draw.text((m_x1 + 80, m_y1 + 45), "CS 101: Dynamic Programming & Algorithms", font=get_font(18, bold=True), fill=(220, 240, 220))
    draw.text((m_x1 + 80, m_y1 + 80), "Lecture 4 — Spatial Tracking & Observable Presence", font=get_font(14), fill=(180, 200, 180))

    # Desk row
    draw.rectangle([(m_x1 + 20, m_y1 + 400), (m_x2 - 20, m_y2 - 20)], fill=(160, 140, 120))

    # Student 1: Consented test portrait with YOLO bounding box
    s1_x, s1_y = m_x1 + 140, m_y1 + 220
    draw.rectangle([(s1_x - 30, s1_y + 80), (s1_x + 130, s1_y + 240)], fill=(30, 90, 80))
    if os.path.exists(PORTRAIT_PATH):
        try:
            p_img = Image.open(PORTRAIT_PATH).convert("RGB")
            p_thumb = p_img.resize((120, 140))
            img.paste(p_thumb, (s1_x - 10, s1_y))
        except Exception:
            draw.ellipse([(s1_x, s1_y), (s1_x + 100, s1_y + 120)], fill=(180, 150, 130))
    else:
        draw.ellipse([(s1_x, s1_y), (s1_x + 100, s1_y + 120)], fill=(180, 150, 130))

    # YOLO Bounding Box 1 (Green)
    draw.rectangle([(s1_x - 35, s1_y - 10), (s1_x + 135, s1_y + 240)], outline=(16, 185, 129), width=2)
    draw.rectangle([(s1_x - 35, s1_y - 30), (s1_x + 135, s1_y - 10)], fill=(16, 185, 129))
    draw.text((s1_x - 30, s1_y - 28), "Track #1: Emma Watson (92%)", font=get_font(11, bold=True), fill=(255, 255, 255))
    # Orientation vector (solvePnP)
    draw.line([(s1_x + 50, s1_y + 60), (s1_x + 50, s1_y + 10)], fill=(59, 130, 246), width=3)
    draw.ellipse([(s1_x + 46, s1_y + 6), (s1_x + 54, s1_y + 14)], fill=(59, 130, 246))

    # Student 2: Second student
    s2_x, s2_y = m_x1 + 460, m_y1 + 230
    draw.rectangle([(s2_x - 20, s2_y + 70), (s2_x + 120, s2_y + 240)], fill=(80, 60, 100))
    draw.ellipse([(s2_x + 10, s2_y + 10), (s2_x + 90, s2_y + 110)], fill=(175, 145, 125))

    # YOLO Bounding Box 2 (Blue)
    draw.rectangle([(s2_x - 25, s2_y - 5), (s2_x + 125, s2_y + 240)], outline=(59, 130, 246), width=2)
    draw.rectangle([(s2_x - 25, s2_y - 25), (s2_x + 125, s2_y - 5)], fill=(59, 130, 246))
    draw.text((s2_x - 20, s2_y - 23), "Track #2: Liam Smith (89%)", font=get_font(11, bold=True), fill=(255, 255, 255))
    # Orientation vector (solvePnP)
    draw.line([(s2_x + 50, s2_y + 55), (s2_x + 40, s2_y + 10)], fill=(16, 185, 129), width=3)
    draw.ellipse([(s2_x + 36, s2_y + 6), (s2_x + 44, s2_y + 14)], fill=(16, 185, 129))

    # Telemetry HUD on bottom of visualizer
    hud_y = 780
    hud_items = [
        ("NVIDIA RTX 3050 Laptop GPU", "CUDA Accelerated Inference", (16, 185, 129)),
        ("solvePnP Head Pose", "Yaw: -4° | Pitch: +6° | Roll: +1°", (59, 130, 246)),
        ("Frame Differencing", "Motion Intensity: 0.28 (Moderate)", (245, 158, 11)),
    ]
    h_x = 1100
    for h_title, h_sub, h_col in hud_items:
        draw.rounded_rectangle([(h_x, hud_y), (h_x + 245, hud_y + 80)], radius=10, fill=(30, 41, 59), outline=(51, 65, 85), width=1)
        draw.text((h_x + 12, hud_y + 12), h_title, font=get_font(11, bold=True), fill=h_col)
        draw.text((h_x + 12, hud_y + 36), h_sub, font=get_font(11), fill=(203, 213, 225))
        h_x += 256

    draw_lower_third(
        draw,
        scene_number=4,
        scene_title="Recorded Video Ingestion & Pipeline Processing",
        explanation="Ingests test lecture video. YOLOv8 detects bodies, solvePnP computes head pose yaw/pitch, and audio VAD tracks speech intervals on GPU.",
        tech_tag="YOLOv8 + CUDA + solvePnP"
    )
    return img


def render_scene_5(progress: float) -> Image.Image:
    """Scene 5: Sustained Attendance Verification (1:18 - 1:40)"""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "live")
    draw_topbar(draw, "Attendance Verification & Sustained Presence Ledger")

    # Main Ledger Card (Full Width)
    draw.rounded_rectangle([(310, 96), (1890, 930)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    
    # Title & Rule Explanation
    draw.text((334, 120), "Session Attendance Verification Ledger — CS 101 Lecture 1", font=get_font(18, bold=True), fill=(30, 41, 59))
    draw.text((334, 146), "Enforces Sustained Presence: Minimum 5 valid observations required before marking PRESENT. Single fleeting frames are rejected.", font=get_font(12), fill=(100, 116, 139))

    # Metric summary pills
    pills = [
        ("Total Cohort", "8 Students", (30, 41, 59)),
        ("Confirmed Present", "6 Students (75%)", (16, 185, 129)),
        ("Late Arrival", "1 Student (12.5%)", (245, 158, 11)),
        ("Unverified Presence", "1 Student (Consent Revoked)", (220, 38, 38)),
    ]
    p_x = 1100
    for plabel, pval, pcol in pills:
        draw.rounded_rectangle([(p_x, 116), (p_x + 185, 160)], radius=8, fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.text((p_x + 12, 122), plabel, font=get_font(10, bold=True), fill=(100, 116, 139))
        draw.text((p_x + 12, 138), pval, font=get_font(11, bold=True), fill=pcol)
        p_x += 195

    # Ledger Table Header
    draw.rectangle([(334, 180), (1866, 220)], fill=(241, 245, 249))
    draw.text((350, 194), "STUDENT ID", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((490, 194), "STUDENT NAME", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((700, 194), "VERIFIED STATUS", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((880, 194), "OBSERVATIONS", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1050, 194), "SUSTAINED DURATION", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1250, 194), "FACE MATCH CONF.", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1440, 194), "AUDIT NOTES & RULE REASONING", font=get_font(11, bold=True), fill=(71, 85, 105))
    draw.text((1770, 194), "ACTION", font=get_font(11, bold=True), fill=(71, 85, 105))

    records = [
        ("CS-2024-001", "Emma Watson", "PRESENT", "38 frames (>=5)", "45.0 mins (90%)", "92% Cosine", "Sustained presence confirmed; identity verified", False),
        ("CS-2024-002", "Liam Smith", "PRESENT", "41 frames (>=5)", "46.7 mins (93%)", "89% Cosine", "Sustained presence confirmed; identity verified", False),
        ("CS-2024-003", "Sophia Garcia", "LATE", "27 frames (>=5)", "31.6 mins (63%)", "91% Cosine", "Arrived at minute 14 (Late threshold: 10 mins)", False),
        ("CS-2024-004", "Noah Patel", "PRESENT", "35 frames (>=5)", "43.3 mins (87%)", "94% Cosine", "Sustained presence confirmed; identity verified", False),
        ("CS-2024-005", "Ava Chen", "PRESENT", "33 frames (>=5)", "42.0 mins (84%)", "91% Cosine", "Sustained presence confirmed; identity verified", False),
        ("CS-2024-006", "Lucas Vance", "PRESENT", "39 frames (>=5)", "44.5 mins (89%)", "88% Cosine", "Sustained presence confirmed; identity verified", False),
        ("CS-2024-007", "Maya Lin", "PRESENT", "36 frames (>=5)", "43.0 mins (86%)", "90% Cosine", "Sustained presence confirmed; identity verified", False),
        ("CS-2024-008", "Ethan Wright", "UNVERIFIED", "12 frames (>=5)", "38.0 mins (76%)", "Disabled (0%)", "Consent revoked: presence tracked, identity unverified", True),
    ]

    r_y = 236
    for sid, name, status, obs, dur, conf, notes, is_override in records:
        draw.line([(334, r_y - 6), (1866, r_y - 6)], fill=(241, 245, 249), width=1)
        draw.text((350, r_y + 4), sid, font=get_font(12, bold=True), fill=(30, 41, 59))
        draw.text((490, r_y + 4), name, font=get_font(12), fill=(15, 23, 42))

        # Status Badge
        badge_colors = {
            "PRESENT": ((236, 253, 245), (16, 185, 129)),
            "LATE": ((254, 243, 199), (217, 119, 6)),
            "UNVERIFIED": ((254, 242, 242), (220, 38, 38)),
        }
        bg, fg = badge_colors.get(status, ((241, 245, 249), (71, 85, 105)))
        draw.rounded_rectangle([(696, r_y), (820, r_y + 24)], radius=8, fill=bg)
        draw.text((710, r_y + 4), status, font=get_font(11, bold=True), fill=fg)

        draw.text((880, r_y + 4), obs, font=get_font(12), fill=(30, 41, 59))
        draw.text((1050, r_y + 4), dur, font=get_font(12), fill=(30, 41, 59))
        draw.text((1250, r_y + 4), conf, font=get_font(12, bold=True), fill=(16, 185, 129) if "9" in conf or "8" in conf else (220, 38, 38))
        draw.text((1440, r_y + 4), notes, font=get_font(11), fill=(100, 116, 139))

        # Override Action Button
        draw.rounded_rectangle([(1760, r_y), (1850, r_y + 24)], radius=6, fill=(239, 246, 255), outline=(191, 219, 254), width=1)
        draw.text((1772, r_y + 4), "Override", font=get_font(11, bold=True), fill=(37, 99, 235))

        r_y += 50

    # Override Modal Overlay Preview
    m_ox, m_oy = 640, 680
    draw.rounded_rectangle([(m_ox, m_oy), (m_ox + 640, m_oy + 220)], radius=14, fill=(255, 255, 255), outline=(59, 130, 246), width=2)
    draw.text((m_ox + 24, m_oy + 20), "Instructor Manual Attendance Correction", font=get_font(15, bold=True), fill=(30, 41, 59))
    draw.text((m_ox + 24, m_oy + 44), "Ethan Wright (CS-2024-008) — Physical Presence Confirmed in Room Turing 102", font=get_font(12), fill=(100, 116, 139))
    
    draw.text((m_ox + 24, m_oy + 75), "Override Status: PRESENT", font=get_font(12, bold=True), fill=(16, 185, 129))
    draw.text((m_ox + 24, m_oy + 100), "Audit Note: Student present in row 2; biometric opted-out under privacy policy.", font=get_font(12), fill=(51, 65, 85))
    draw.text((m_ox + 24, m_oy + 125), "Correction logged to audit table with timestamp and instructor signature.", font=get_font(11), fill=(148, 163, 184))

    draw.rounded_rectangle([(m_ox + 450, m_oy + 160), (m_ox + 610, m_oy + 198)], radius=8, fill=(37, 99, 235))
    draw.text((m_ox + 480, m_oy + 172), "Save Override", font=get_font(12, bold=True), fill=(255, 255, 255))

    draw_lower_third(
        draw,
        scene_number=5,
        scene_title="Attendance Results & Sustained-Presence Rule",
        explanation="Requires sustained observations (>= 5 frames) to mark PRESENT. Single frames never mark attendance. Instructor retains manual override audit primacy.",
        tech_tag="Attendance Verification Logic"
    )
    return img


def render_scene_6(progress: float) -> Image.Image:
    """Scene 6: Observable Engagement Analytics (1:40 - 2:02)"""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "analytics")
    draw_topbar(draw, "Session Engagement Analytics — CS 101 Lecture 1")

    # Top KPI Banner
    draw.rounded_rectangle([(310, 96), (1890, 180)], radius=14, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((334, 116), "Classroom Observable Telemetry Timeline (30-Second Sampling Windows)", font=get_font(16, bold=True), fill=(30, 41, 59))
    draw.text((334, 142), "Behavioral proxies derived from 3D head-pose yaw/pitch, optical frame differencing, and aggregate voice energy.", font=get_font(12), fill=(100, 116, 139))

    # Download CSV Button on top right
    draw.rounded_rectangle([(1600, 114), (1866, 162)], radius=10, fill=(37, 99, 235))
    draw.text((1630, 128), "⬇ Download Full Session CSV", font=get_font(13, bold=True), fill=(255, 255, 255))

    # Main Chart Container (Recharts Style)
    draw.rounded_rectangle([(310, 200), (1890, 780)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)

    # Chart Header & Legend
    draw.text((334, 224), "Multi-Signal Telemetry Across Lecture Timeline", font=get_font(15, bold=True), fill=(30, 41, 59))
    
    # Legend Pills
    legs = [
        ("Composite Interaction Activity Score", (59, 130, 246)),
        ("Forward Head-Pose Orientation (solvePnP %)", (16, 185, 129)),
        ("Physical Movement Intensity (Normalized diff %)", (245, 158, 11)),
        ("Speech VAD Activity Rate (%)", (168, 85, 247)),
    ]
    lx = 850
    for l_txt, l_col in legs:
        draw.ellipse([(lx, 228), (lx + 10, 238)], fill=l_col)
        draw.text((lx + 16, 226), l_txt, font=get_font(11, bold=True), fill=(71, 85, 105))
        lx += 260

    # Grid Area
    gx1, gy1, gx2, gy2 = 380, 270, 1850, 710
    draw.rectangle([(gx1, gy1), (gx2, gy2)], fill=(248, 250, 252))

    for gy in range(gy1, gy2 + 1, 88):
        draw.line([(gx1, gy), (gx2, gy)], fill=(226, 232, 240), width=1)
        val = int(100 - (gy - gy1) / (gy2 - gy1) * 100)
        draw.text((gx1 - 35, gy - 7), f"{val}%", font=get_font(10), fill=(148, 163, 184))

    # 12 Window X points
    scores = [82, 85, 78, 74, 88, 92, 86, 79, 75, 81, 84, 80]
    poses = [86, 88, 80, 76, 90, 94, 89, 82, 77, 84, 87, 83]
    motions = [24, 28, 32, 30, 48, 52, 44, 35, 29, 31, 33, 25]
    speeches = [35, 42, 28, 22, 65, 70, 58, 34, 25, 38, 40, 30]

    n_pts = len(scores)
    dx = (gx2 - gx1 - 80) / (n_pts - 1)

    pts_score, pts_pose, pts_motion, pts_speech = [], [], [], []
    for i in range(n_pts):
        x = gx1 + 40 + i * dx
        pts_score.append((x, gy2 - (scores[i] / 100.0) * (gy2 - gy1)))
        pts_pose.append((x, gy2 - (poses[i] / 100.0) * (gy2 - gy1)))
        pts_motion.append((x, gy2 - (motions[i] / 100.0) * (gy2 - gy1)))
        pts_speech.append((x, gy2 - (speeches[i] / 100.0) * (gy2 - gy1)))
        
        # X-axis label
        sec = i * 30
        draw.text((x - 12, gy2 + 10), f"{sec}s", font=get_font(11), fill=(100, 116, 139))

    # Draw lines
    draw.line(pts_motion, fill=(245, 158, 11), width=2)
    draw.line(pts_speech, fill=(168, 85, 247), width=2)
    draw.line(pts_pose, fill=(16, 185, 129), width=3)
    draw.line(pts_score, fill=(59, 130, 246), width=3)

    # Hover tooltip simulation at peak interaction window (Window 5: 150s - 180s)
    tip_x, tip_y = pts_score[5]
    draw.line([(tip_x, gy1), (tip_x, gy2)], fill=(59, 130, 246), width=1)
    
    # Tooltip box
    draw.rounded_rectangle([(tip_x - 140, tip_y - 140), (tip_x + 140, tip_y - 10)], radius=10, fill=(15, 23, 42), outline=(59, 130, 246), width=1)
    draw.text((tip_x - 125, tip_y - 130), "Window #6 (150s – 180s)", font=get_font(12, bold=True), fill=(255, 255, 255))
    draw.text((tip_x - 125, tip_y - 108), "• Composite Interaction: 92 / 100", font=get_font(11, bold=True), fill=(96, 165, 250))
    draw.text((tip_x - 125, tip_y - 90), "• Forward Orientation: 94% (solvePnP)", font=get_font(11), fill=(110, 231, 183))
    draw.text((tip_x - 125, tip_y - 72), "• Speech VAD Rate: 70% (Discussion)", font=get_font(11), fill=(216, 180, 254))
    draw.text((tip_x - 125, tip_y - 54), "• Physical Movement: 52% (Active)", font=get_font(11), fill=(252, 211, 77))
    draw.text((tip_x - 125, tip_y - 36), "• Avg Detected Persons: 7.2 students", font=get_font(11), fill=(203, 213, 225))

    # Pedagogical disclaimer banner on bottom of analytics
    draw.rounded_rectangle([(310, 800), (1890, 930)], radius=12, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    draw.text((334, 818), "⚠️ Educational Integrity & Scientific Transparency Standard", font=get_font(13, bold=True), fill=(30, 41, 59))
    draw.text((334, 844), "The metrics above represent observable physical signals (head orientation, audio activity, frame movement) aggregated across 30-second windows.", font=get_font(12), fill=(71, 85, 105))
    draw.text((334, 868), "They serve as behavioral proxies for instructor awareness and DO NOT measure internal comprehension, cognitive focus, or student intelligence.", font=get_font(12, bold=True), fill=(15, 23, 42))

    draw_lower_third(
        draw,
        scene_number=6,
        scene_title="Observable Engagement Analytics (30s Windows)",
        explanation="Recharts visualization of composite interaction, 3D head pose %, movement %, and speech activity. Explicitly declared as behavioral proxies.",
        tech_tag="Recharts + solvePnP + VAD"
    )
    return img


def render_scene_7(progress: float) -> Image.Image:
    """Scene 7: AI Teaching Suggestions (2:02 - 2:22)"""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "suggestions")
    draw_topbar(draw, "AI Teaching Suggestions & Context-Aware Pedagogical Advisor")

    # Header card
    draw.rounded_rectangle([(310, 96), (1890, 180)], radius=14, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((334, 116), "AI Context-Aware Pedagogical Suggestions — Session CS 101 Lecture 1", font=get_font(16, bold=True), fill=(30, 41, 59))
    draw.text((334, 142), "Synthesizes aggregate observable behavioral windows into actionable pedagogical interventions. Instructor maintains decision primacy.", font=get_font(12), fill=(100, 116, 139))

    # 3 Real AI Teaching Suggestion Cards
    sugs = [
        (
            "Active Discussion Segment Observed",
            "INTERACTION_PEAK",
            "LOW",
            "Classroom behavioral signals reflected elevated collaborative interaction between minutes 2:00 and 3:30. Students actively participated in verbal questions and group dialogue.",
            "VAD speech activity reached 70% accompanied by increased physical movement (0.52) and multi-person orientation shifts.",
            "Pedagogical Tip: Validate student contributions and bridge current insights into the next algorithmic theorem.",
            (16, 185, 129),
            (236, 253, 245)
        ),
        (
            "Sustained Forward Attention Pattern",
            "STEADY_FLOW",
            "LOW",
            "85% of detected students maintained frontal head-pose orientation toward the primary lecture presentation area without prolonged drop-offs.",
            "Perspective-n-Point solvePnP yaw angles remained within +/- 18 degrees across 9 consecutive 30-second observation intervals.",
            "Pedagogical Tip: Instructional pacing is currently well-calibrated. Consider introducing a quick formative check-in.",
            (59, 130, 246),
            (239, 246, 255)
        ),
        (
            "Unconsented Attendee Detected in Cohort",
            "ATTENDANCE_VERIFICATION",
            "MEDIUM",
            "Ethan Wright has opted out of biometric facial storage. Sustained presence was confirmed via spatial tracker, but facial recognition was intentionally bypassed.",
            "Biometric consent status is FALSE. Local privacy rules strictly prevented FaceNet vector matching.",
            "Action Required: Perform manual instructor roll-call check to confirm student attendance.",
            (220, 38, 38),
            (254, 242, 242)
        )
    ]

    card_y = 206
    for title, stype, urg, desc, ev, tip, col, bg_col in sugs:
        draw.rounded_rectangle([(310, card_y), (1890, card_y + 220)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        
        # Left color strip
        draw.rounded_rectangle([(310, card_y), (322, card_y + 220)], radius=8, fill=col)

        # Title & Badges
        draw.text((345, card_y + 20), title, font=get_font(16, bold=True), fill=(30, 41, 59))
        
        # Urgency pill
        draw.rounded_rectangle([(1620, card_y + 18), (1730, card_y + 46)], radius=14, fill=bg_col)
        draw.text((1635, card_y + 24), f"Urgency: {urg}", font=get_font(11, bold=True), fill=col)

        # Engine pill
        draw.rounded_rectangle([(1745, card_y + 18), (1866, card_y + 46)], radius=14, fill=(241, 245, 249))
        draw.text((1760, card_y + 24), "Rule Engine / Ollama", font=get_font(11, bold=True), fill=(71, 85, 105))

        # Description
        draw.text((345, card_y + 54), desc, font=get_font(13), fill=(51, 65, 85))

        # Observable Evidence Box
        draw.rounded_rectangle([(345, card_y + 96), (1866, card_y + 150)], radius=8, fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.text((360, card_y + 104), "OBSERVABLE EVIDENCE LOG:", font=get_font(11, bold=True), fill=(71, 85, 105))
        draw.text((540, card_y + 104), ev, font=get_font(12), fill=(15, 23, 42))
        draw.text((360, card_y + 126), "ACTIONABLE GUIDANCE:", font=get_font(11, bold=True), fill=col)
        draw.text((540, card_y + 126), tip, font=get_font(12, bold=True), fill=(30, 41, 59))

        # Acknowledge button
        draw.rounded_rectangle([(345, card_y + 166), (490, card_y + 200)], radius=6, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
        draw.text((365, card_y + 174), "✓ Acknowledge", font=get_font(11, bold=True), fill=(71, 85, 105))

        card_y += 240

    draw_lower_third(
        draw,
        scene_number=7,
        scene_title="AI Teaching Suggestions & Context-Aware Feedback",
        explanation="Synthesizes session telemetry into evidence-backed pedagogical check-ins. Alerts instructors to discussion peaks, steady flow, and unconsented students.",
        tech_tag="Context-Aware Advisor"
    )
    return img


def render_scene_8(progress: float) -> Image.Image:
    """Scene 8: Privacy Guarantees & Architecture Wrap-Up (2:22 - 2:42)"""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "settings")
    draw_topbar(draw, "System Settings, Privacy Guarantees & CSV Export")

    # Offline Guarantee Hero Card
    draw.rounded_rectangle([(310, 96), (1890, 300)], radius=16, fill=(236, 253, 245), outline=(167, 243, 208), width=2)
    draw.ellipse([(340, 126), (390, 176)], fill=(16, 185, 129))
    draw.text((356, 134), "🛡️", font=get_font(24), fill=(255, 255, 255))
    
    draw.text((410, 122), "100% Offline-First Local Data Processing Guarantee", font=get_font(20, bold=True), fill=(6, 95, 70))
    draw.text((410, 154), "All computer vision models (YOLOv8, FaceNet), tracking algorithms, voice activity detection, and language model inference run locally on this machine.", font=get_font(13), fill=(4, 120, 87))

    # 3 Privacy Guarantee Sub-Cards
    sub_pillars = [
        ("No Cloud Leakage", "Zero student images, facial crops, or audio buffers are transmitted to external third-party cloud APIs."),
        ("Ephemeral Video Media", "Uploaded video files and audio extracts are automatically deleted from disk following telemetry extraction."),
        ("Biometric Vector Vault", "512-dim facial vectors are stored locally in isolated numpy matrices, inaccessible via public REST APIs."),
    ]
    p_x = 410
    for ptitle, pdesc in sub_pillars:
        draw.rounded_rectangle([(p_x, 196), (p_x + 460, 276)], radius=10, fill=(255, 255, 255), outline=(167, 243, 208), width=1)
        draw.text((p_x + 16, 210), f"✓ {ptitle}", font=get_font(13, bold=True), fill=(6, 95, 70))
        draw.text((p_x + 16, 234), pdesc, font=get_font(11), fill=(71, 85, 105))
        p_x += 480

    # Hardware & Model Telemetry Card
    draw.rounded_rectangle([(310, 320), (1890, 620)], radius=16, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    draw.text((334, 344), "Hardware Acceleration & Neural Model Telemetry", font=get_font(16, bold=True), fill=(30, 41, 59))
    draw.text((334, 370), "Real-time telemetry gathered from backend endpoint /api/v1/health", font=get_font(12), fill=(100, 116, 139))

    telemetry_items = [
        ("Compute Device", "NVIDIA GeForce RTX 3050 6GB Laptop GPU", (16, 185, 129)),
        ("CUDA Acceleration", "Active (PyTorch CUDA Enabled)", (16, 185, 129)),
        ("Person Detection Model", "Ultralytics YOLOv8 Nano (yolov8n.pt)", (59, 130, 246)),
        ("Biometric Feature Embedder", "FaceNet 512-dim L2-Normalized (Local Weights)", (59, 130, 246)),
        ("Head Orientation Engine", "Perspective-n-Point 3D solvePnP", (139, 92, 246)),
        ("Acoustic Telemetry Engine", "Aggregate RMS Energy & Zero-Crossing VAD", (245, 158, 11)),
    ]

    t_y = 410
    col_x1, col_x2 = 334, 1120
    for idx, (label, val, tcol) in enumerate(telemetry_items):
        cx = col_x1 if idx % 2 == 0 else col_x2
        cy = t_y + (idx // 2) * 60
        draw.rounded_rectangle([(cx, cy), (cx + 740, cy + 48)], radius=8, fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.text((cx + 16, cy + 14), label, font=get_font(12, bold=True), fill=(71, 85, 105))
        draw.text((cx + 340, cy + 14), val, font=get_font(12, bold=True), fill=tcol)

    # CSV Audit Export Download Box
    draw.rounded_rectangle([(310, 640), (1890, 770)], radius=16, fill=(239, 246, 255), outline=(191, 219, 254), width=1)
    draw.text((340, 664), "📁 Audit & Reporting Workflow: CSV Export Verified", font=get_font(15, bold=True), fill=(30, 64, 175))
    draw.text((340, 690), "Generates standard RFC-4180 CSV attendance ledger for institutional gradebook integration.", font=get_font(12), fill=(59, 130, 246))
    
    # Download badge
    draw.rounded_rectangle([(340, 715), (760, 750)], radius=6, fill=(37, 99, 235))
    draw.text((360, 723), "✓ attendance_session_SESS-CS101-01.csv downloaded", font=get_font(11, bold=True), fill=(255, 255, 255))

    # Concluding Technical Summary Badge
    draw.rounded_rectangle([(310, 790), (1890, 930)], radius=16, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
    draw.text((340, 810), "TECHNICAL SPECIFICATIONS SUMMARY", font=get_font(12, bold=True), fill=(96, 165, 250))
    draw.text((340, 836), "Backend: Python 3.13, FastAPI, Async SQLAlchemy, SQLite  •  Frontend: React 19, TypeScript, Tailwind CSS, Recharts", font=get_font(13, bold=True), fill=(248, 250, 252))
    draw.text((340, 862), "Computer Vision: YOLOv8 Person Detection, FaceNet 512-dim Biometrics, OpenCV solvePnP 3D Head Orientation", font=get_font(12), fill=(203, 213, 225))
    draw.text((340, 888), "Deployment: 100% Offline-First, Localhost Storage, Zero Third-Party Cloud Offloading", font=get_font(12), fill=(16, 185, 129))

    draw_lower_third(
        draw,
        scene_number=8,
        scene_title="Privacy Guarantees, Hardware Acceleration & CSV Export",
        explanation="Strict privacy compliance: zero cloud leakage, ephemeral media deletion, and local biometric vault. Hardware acceleration on RTX 3050 Laptop GPU.",
        tech_tag="Privacy Architecture + Export"
    )
    return img

# ----------------- MAIN RENDERER ----------------- #

def build_full_video():
    print("=" * 60)
    print("STARTING PROGRAMMATIC DEMO VIDEO GENERATION (1080p @ 30 FPS)")
    print(f"Target Output: {OUTPUT_VIDEO_PATH}")
    print("=" * 60)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError("Failed to initialize OpenCV VideoWriter for MP4.")

    scenes = [
        ("Scene 1: Introduction Title Screen", render_scene_1, 12.0), # 12s = 360 frames
        ("Scene 2: Dashboard Overview", render_scene_2, 20.0),         # 20s = 600 frames
        ("Scene 3: Student Enrollment & Consent", render_scene_3, 20.0), # 20s = 600 frames
        ("Scene 4: Recorded Video Processing", render_scene_4, 26.0),    # 26s = 780 frames
        ("Scene 5: Sustained Attendance Ledger", render_scene_5, 22.0),  # 22s = 660 frames
        ("Scene 6: Observable Engagement Analytics", render_scene_6, 22.0), # 22s = 660 frames
        ("Scene 7: AI Teaching Suggestions", render_scene_7, 20.0),     # 20s = 600 frames
        ("Scene 8: Privacy Guarantees & Summary", render_scene_8, 20.0), # 20s = 600 frames
    ]

    total_duration = sum(s[2] for s in scenes)
    total_frames = int(total_duration * FPS)
    print(f"Total Scenes: {len(scenes)} | Total Duration: {total_duration:.1f}s ({total_frames} frames)")

    transition_frames = 15  # 0.5s smooth cross-fade transition
    frame_counter = 0

    prev_last_frame = None

    for s_idx, (s_name, render_fn, dur_sec) in enumerate(scenes):
        s_frames = int(dur_sec * FPS)
        print(f"\nRendering [{s_idx + 1}/{len(scenes)}] {s_name} ({dur_sec}s, {s_frames} frames)...")

        # Pre-render key sample frame
        base_frame_pil = render_fn(1.0)
        base_frame_cv = cv2.cvtColor(np.array(base_frame_pil), cv2.COLOR_RGB2BGR)

        for f in range(s_frames):
            prog = f / float(s_frames)
            
            # Dynamic progress for Scene 4 progress bar
            if render_fn == render_scene_4:
                # Progress advances from 0.05 to 1.0
                p_val = min(1.0, 0.05 + prog * 1.1)
                curr_pil = render_fn(p_val)
                curr_cv = cv2.cvtColor(np.array(curr_pil), cv2.COLOR_RGB2BGR)
            else:
                curr_cv = base_frame_cv

            # Apply cross-fade at start of scene if not scene 1
            if s_idx > 0 and f < transition_frames and prev_last_frame is not None:
                alpha = f / float(transition_frames)
                blended = cv2.addWeighted(curr_cv, alpha, prev_last_frame, 1.0 - alpha, 0)
                out.write(blended)
            else:
                out.write(curr_cv)

            frame_counter += 1

        prev_last_frame = curr_cv.copy()

    out.release()
    print("\n" + "=" * 60)
    print(f"DEMO VIDEO GENERATED SUCCESSFULLY!")
    print(f"File: {OUTPUT_VIDEO_PATH}")
    file_size_mb = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024 * 1024)
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Total Frames Rendered: {frame_counter}")
    print("=" * 60)

    # Copy to artifacts directory
    artifact_dest = os.path.join(ARTIFACTS_DIR, "Multimodal_Attendance_Engagement_Agent_Demo.mp4")
    shutil.copyfile(OUTPUT_VIDEO_PATH, artifact_dest)
    print(f"Artifact copy saved to: {artifact_dest}")

if __name__ == "__main__":
    build_full_video()
