# utils.py — Angle calculation, distance helpers, drawing utilities

import cv2
import numpy as np
import time
import threading
try:
    import winsound
except ImportError:
    winsound = None

from config import *

# ── Audio Cues ─────────────────────────────────────────────────────────────

def play_phase_sound(phase_type):
    """Play a non-blocking sound for phase transitions."""
    if not winsound:
        return
    def beep():
        if phase_type == "HOLD":
            winsound.Beep(700, 300)  # Higher pitch for entering hold
        elif phase_type == "REST":
            winsound.Beep(400, 300)  # Lower pitch for rest
        elif phase_type == "COMPLETE":
            winsound.Beep(600, 150)
            winsound.Beep(800, 300)

    threading.Thread(target=beep, daemon=True).start()



# ── Angle & Distance ───────────────────────────────────────────────────────

def calculate_angle(A, B, C):
    """
    Angle at joint B given three landmarks A (proximal), B (vertex), C (distal).
    Uses dot product formula: arccos(BA·BC / |BA||BC|)
    Returns degrees in [0, 180]. Returns None if any point is invalid.
    """
    a = np.array([A.x, A.y])
    b = np.array([B.x, B.y])
    c = np.array([C.x, C.y])

    ba = a - b
    bc = c - b

    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return None

    cosine = np.dot(ba, bc) / norm
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def landmark_visible(lm, threshold=MIN_VISIBILITY):
    """Returns True if landmark has sufficient confidence."""
    return lm.visibility >= threshold


def pixel_distance(lm1, lm2, frame_w, frame_h):
    """Euclidean pixel distance between two landmarks."""
    x1, y1 = lm1.x * frame_w, lm1.y * frame_h
    x2, y2 = lm2.x * frame_w, lm2.y * frame_h
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def world_distance_cm(wlm1, wlm2):
    """
    3D Euclidean distance between two world landmarks (in metres → cm).
    World landmarks are in metric units relative to hip midpoint.
    """
    dx = wlm1.x - wlm2.x
    dy = wlm1.y - wlm2.y
    dz = wlm1.z - wlm2.z
    return np.sqrt(dx**2 + dy**2 + dz**2) * 100  # metres → cm


# ── FPS Counter ────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self):
        self._prev = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        self.fps = 1.0 / max(now - self._prev, 1e-6)
        self._prev = now
        return self.fps


# ── Drawing Helpers ────────────────────────────────────────────────────────

def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.6):
    """Semi-transparent rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_status_panel(frame, checks, hold_timer, rest_timer, phase, fps, pose_name="urdhva"):
    """
    Draws the right-side status panel showing all 6 checks,
    breath hold progress bar, and FPS.
    """
    h, w = frame.shape[:2]
    panel_x = w - 310
    panel_w = 300

    # Panel background
    draw_rounded_rect(frame, panel_x - 5, 10, panel_w, h - 20, 8, (30, 30, 30), alpha=0.55)

    # Title
    cv2.putText(frame, "POSE CHECKS", (panel_x + 55, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

    if pose_name == "warrior2":
        labels = [
            "1. Stance wide",
            "2. Arms horizontal",
            "3. Front knee bent",
            "4. Back leg straight",
            f"5. Hold {WARRIOR_HOLD_BREATHS} breaths",
            "6. Rest phase",
        ]
        hold_duration = WARRIOR_HOLD_DURATION
        hold_breaths = WARRIOR_HOLD_BREATHS
        rest_duration = WARRIOR_REST_DURATION
        rest_breaths = WARRIOR_REST_BREATHS
    elif pose_name == "warrior1":
        labels = [
            "1. Stance wide",
            "2. Arms raised & palms joined",
            "3. Front knee bent",
            "4. Back leg straight",
            f"5. Hold {WARRIOR1_HOLD_BREATHS} breaths",
            "6. Rest phase",
        ]
        hold_duration = WARRIOR1_HOLD_DURATION
        hold_breaths = WARRIOR1_HOLD_BREATHS
        rest_duration = WARRIOR1_REST_DURATION
        rest_breaths = WARRIOR1_REST_BREATHS
    elif pose_name == "warrior3":
        labels = [
            "1. Standing leg straight",
            "2. Raised leg straight",
            "3. Torso & leg parallel",
            "4. Arms extended forward",
            f"5. Hold {WARRIOR3_HOLD_BREATHS} breaths",
            "6. Rest phase",
        ]
        hold_duration = WARRIOR3_HOLD_DURATION
        hold_breaths = WARRIOR3_HOLD_BREATHS
        rest_duration = WARRIOR3_REST_DURATION
        rest_breaths = WARRIOR3_REST_BREATHS
    else:
        labels = [
            "1. Feet 15cm apart",
            "2. Arms stretched",
            "3. Arms near ears",
            "4. Palms joined",
            f"5. Hold {HOLD_BREATHS} breaths",
            "6. Rest phase",
        ]
        hold_duration = HOLD_DURATION
        hold_breaths = HOLD_BREATHS
        rest_duration = REST_DURATION
        rest_breaths = REST_BREATHS

    for i, (label, ok) in enumerate(zip(labels, checks)):
        y = 70 + i * 38
        icon = "✓" if ok else "✗"
        color = COLOR_GREEN if ok else COLOR_RED
        cv2.putText(frame, f"{icon} {label}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Hold progress bar
    bar_y = 310
    cv2.putText(frame, "Hold Progress:", (panel_x, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1)
    bar_fill = int(min(hold_timer / hold_duration, 1.0) * (panel_w - 20))
    cv2.rectangle(frame, (panel_x, bar_y + 8), (panel_x + panel_w - 20, bar_y + 24),
                  (80, 80, 80), -1)
    cv2.rectangle(frame, (panel_x, bar_y + 8), (panel_x + bar_fill, bar_y + 24),
                  COLOR_GREEN, -1)
    breaths_done = min(int(hold_timer / BREATH_DURATION), hold_breaths)
    cv2.putText(frame, f"{breaths_done}/{hold_breaths} breaths  ({hold_timer:.1f}s)",
                (panel_x, bar_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # Rest progress bar
    rest_y = bar_y + 65
    cv2.putText(frame, "Rest Progress:", (panel_x, rest_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1)
    rest_fill = int(min(rest_timer / rest_duration, 1.0) * (panel_w - 20))
    cv2.rectangle(frame, (panel_x, rest_y + 8), (panel_x + panel_w - 20, rest_y + 24),
                  (80, 80, 80), -1)
    cv2.rectangle(frame, (panel_x, rest_y + 8), (panel_x + rest_fill, rest_y + 24),
                  COLOR_BLUE, -1)
    rest_done = min(int(rest_timer / BREATH_DURATION), rest_breaths)
    cv2.putText(frame, f"{rest_done}/{rest_breaths} breaths  ({rest_timer:.1f}s)",
                (panel_x, rest_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # Phase indicator
    phase_y = rest_y + 68
    phase_color = COLOR_YELLOW if phase == "HOLD" else COLOR_ORANGE
    cv2.putText(frame, f"Phase: {phase}", (panel_x, phase_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_angle_label(frame, lm, angle, w, h):
    """Draw angle value near a joint landmark."""
    if angle is None:
        return
    x = int(lm.x * w)
    y = int(lm.y * h)
    cv2.putText(frame, f"{int(angle)}", (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1, cv2.LINE_AA)


def draw_feedback_banner(frame, message, color):
    """Large centered feedback text at top of frame."""
    h, w = frame.shape[:2]
    draw_rounded_rect(frame, 10, 10, w - 330, 50, 6, (20, 20, 20), alpha=0.6)
    cv2.putText(frame, message, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_coordinates(frame, landmarks, w, h):
    """Print coordinates of key joints in bottom-left corner."""
    key_ids = [11, 12, 13, 14, 15, 16, 23, 24, 27, 28]
    names   = ["LSho","RSho","LElb","RElb","LWri","RWri","LHip","RHip","LAnk","RAnk"]
    draw_rounded_rect(frame, 5, h - 230, 200, 220, 6, (20, 20, 20), alpha=0.5)
    cv2.putText(frame, "Landmark Coords", (10, h - 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
    for i, (idx, name) in enumerate(zip(key_ids, names)):
        lm = landmarks[idx]
        text = f"{name}: ({lm.x:.2f},{lm.y:.2f})"
        cv2.putText(frame, text, (10, h - 195 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)
