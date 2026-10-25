import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = r"C:\Windows\Fonts"
FONT_REG = os.path.join(FONTS_DIR, "segoeui.ttf")
FONT_BOLD = os.path.join(FONTS_DIR, "segoeuib.ttf")
FONT_MONO = os.path.join(FONTS_DIR, "consola.ttf")

def get_font(size: int, bold: bool = False, mono: bool = False):
    f_path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    try:
        return ImageFont.truetype(f_path, size)
    except Exception:
        return ImageFont.load_default()

def draw_topbar(draw, active_title: str):
    # Top navbar
    draw.rectangle([(280, 0), (1920, 72)], fill=(255, 255, 255))
    draw.line([(280, 72), (1920, 72)], fill=(226, 232, 240), width=1)
    
    # Title
    draw.text((310, 22), active_title, font=get_font(20, bold=True), fill=(30, 41, 59))
    
    # Status badges on right
    # Offline badge
    draw.rounded_rectangle([(1450, 18), (1600, 52)], radius=17, fill=(236, 253, 245), outline=(167, 243, 208), width=1)
    draw.ellipse([(1465, 31), (1473, 39)], fill=(16, 185, 129))
    draw.text((1482, 26), "100% Offline", font=get_font(13, bold=True), fill=(6, 95, 70))
    
    # GPU badge
    draw.rounded_rectangle([(1615, 18), (1890, 52)], radius=17, fill=(239, 246, 255), outline=(191, 219, 254), width=1)
    draw.ellipse([(1630, 31), (1638, 39)], fill=(59, 130, 246))
    draw.text((1648, 26), "RTX 3050 GPU (CUDA Active)", font=get_font(13, bold=True), fill=(30, 64, 175))

def draw_sidebar(draw, active_tab: str):
    # Sidebar background
    draw.rectangle([(0, 0), (280, 1080)], fill=(15, 23, 42))
    draw.line([(280, 0), (280, 1080)], fill=(30, 41, 59), width=1)
    
    # Brand logo & title
    draw.rounded_rectangle([(24, 20), (60, 56)], radius=10, fill=(37, 99, 235))
    draw.text((33, 23), "MA", font=get_font(18, bold=True), fill=(255, 255, 255))
    draw.text((72, 22), "Multimodal Agent", font=get_font(16, bold=True), fill=(248, 250, 252))
    draw.text((72, 42), "Classroom Intelligence", font=get_font(11), fill=(148, 163, 184))
    
    draw.line([(20, 72), (260, 72)], fill=(30, 41, 59), width=1)
    
    # Nav items
    nav_items = [
        ("dashboard", "Dashboard Overview"),
        ("students", "Student Directory"),
        ("enrollment", "Student & Face Enrollment"),
        ("video", "Recorded Video Ingestion"),
        ("history", "Classroom Session History"),
        ("analytics", "Session Engagement Analytics"),
        ("suggestions", "AI Teaching Suggestions"),
        ("live", "Live Attendance Tracking"),
        ("settings", "System Settings & Privacy"),
    ]
    
    y = 90
    for tab_id, label in nav_items:
        is_active = (tab_id == active_tab)
        if is_active:
            draw.rounded_rectangle([(16, y), (264, y + 42)], radius=8, fill=(37, 99, 235))
            draw.text((32, y + 11), label, font=get_font(13, bold=True), fill=(255, 255, 255))
        else:
            draw.text((32, y + 11), label, font=get_font(13), fill=(148, 163, 184))
        y += 48
        
    # Footer privacy note in sidebar
    draw.line([(20, 960), (260, 960)], fill=(30, 41, 59), width=1)
    draw.text((24, 980), "Local Processing Vault", font=get_font(12, bold=True), fill=(226, 232, 240))
    draw.text((24, 1000), "Zero Cloud Dependencies", font=get_font(11), fill=(100, 116, 139))
    draw.text((24, 1018), "Ephemeral Media Retention", font=get_font(11), fill=(100, 116, 139))

def draw_lower_third(draw, scene_number: int, scene_title: str, explanation: str, tech_tag: str):
    # Professional translucent lower-third banner
    y1, y2 = 960, 1060
    draw.rectangle([(290, y1), (1910, y2)], fill=(15, 23, 42))
    draw.rounded_rectangle([(290, y1), (1910, y2)], radius=12, fill=(15, 23, 42), outline=(51, 65, 85), width=1)
    
    # Left accent indicator
    draw.rounded_rectangle([(302, y1 + 12), (307, y2 - 12)], radius=3, fill=(59, 130, 246))
    
    # Scene Title
    header = f"Scene {scene_number} • {scene_title}"
    draw.text((322, y1 + 14), header, font=get_font(17, bold=True), fill=(255, 255, 255))
    
    # Explanation
    draw.text((322, y1 + 44), explanation, font=get_font(13), fill=(203, 213, 225))
    
    # Tech tag pill on right
    draw.rounded_rectangle([(1550, y1 + 16), (1890, y1 + 48)], radius=16, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    draw.text((1565, y1 + 22), f"Module: {tech_tag}", font=get_font(12, bold=True), fill=(96, 165, 250))

print("Helper definitions loaded.")
