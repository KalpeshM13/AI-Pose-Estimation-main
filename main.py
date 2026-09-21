# main.py — Yoga Pose Correction Engine
# Exercise: Urdhva Hastasana (Raised Hands Pose)
# Run with: python3.11 main.py

import cv2
import time

from pose_engine import create_pose_detector, process_frame, get_landmarks, get_world_landmarks, open_webcam
from correction import evaluate_pose, autodetect_pose
from utils import (FPSCounter, draw_status_panel, draw_angle_label,
                   draw_feedback_banner, draw_coordinates, calculate_angle, play_phase_sound)
from config import *
from collections import deque, Counter

import argparse

def get_primary_feedback(checks, messages, phase):
    """Return the most important feedback message to show in the banner."""
    if phase == "COMPLETE":
        return "Set Complete! Lower arms and rest.", COLOR_GREEN
    if phase == "REST":
        return messages[5], COLOR_ORANGE
    if phase == "HOLD":
        if all(checks[:4]):
            return messages[4], COLOR_GREEN
        # Show first failing spatial check
        for i in range(4):
            if not checks[i]:
                return messages[i], COLOR_RED
    # SETUP phase — guide user into position
    for i in range(4):
        if not checks[i]:
            return messages[i], COLOR_RED
    return "Great posture! Hold it.", COLOR_GREEN


def get_pose_metadata(pose_id):
    if pose_id == "warrior2":
        return "Virabhadrasana II (Warrior II)", WARRIOR_HOLD_DURATION, WARRIOR_REST_DURATION, 3
    elif pose_id == "warrior1":
        return "Virabhadrasana I (Warrior I)", WARRIOR1_HOLD_DURATION, WARRIOR1_REST_DURATION, 3
    elif pose_id == "warrior3":
        return "Virabhadrasana III (Warrior III)", WARRIOR3_HOLD_DURATION, WARRIOR3_REST_DURATION, 3
    else:
        return "Urdhva Hastasana (Raised Hands Pose)", HOLD_DURATION, REST_DURATION, 3


def main():
    parser = argparse.ArgumentParser(description="Yoga Pose Correction Engine")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (0 for built-in, 1 or 2 for Phone Link/external cameras)")
    parser.add_argument("--pose", type=str, choices=["auto", "urdhva", "warrior1", "warrior2", "warrior3"], default="auto", help="Pose to practice (auto, urdhva, warrior1, warrior2, or warrior3)")
    args = parser.parse_args()

    active_pose = "urdhva" if args.pose == "auto" else args.pose
    pose_history = deque(maxlen=15)
    pose_name_ui, pose_hold_duration, pose_rest_duration, target_reps = get_pose_metadata(active_pose)

    print("=" * 55)
    print("  Yoga Pose Correction Engine")
    print(f"  Exercise Mode: {args.pose.upper()}")
    print("=" * 55)
    print("Controls: Q = quit | R = reset rep counter")
    print(f"Using Camera Index: {args.cam} (Use --cam <index> to change)")
    print()

    cap = open_webcam(cam_index=args.cam, width=1280, height=720)
    fps_counter = FPSCounter()

    # State
    hold_timer = 0.0
    rest_timer = 0.0
    rep_count = 0
    phase = "SETUP"     # SETUP → HOLD → REST → COMPLETE → SETUP
    in_hold_phase = False
    set_complete = False
    prev_time = time.time()

    pose = create_pose_detector()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Flip for mirror view (more intuitive for user)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = fps_counter.update()

            # ── Detection ─────────────────────────────────────────────────
            results = process_frame(pose, frame)
            landmarks = get_landmarks(results)
            world_landmarks = get_world_landmarks(results)

            # Default checks (all False) when no person detected
            checks = [False] * 6
            messages = [""] * 6
            spatial_ok = False
            arms_down = False

            if landmarks:
                if args.pose == "auto" and rep_count == 0:
                    detected = autodetect_pose(landmarks)
                    pose_history.append(detected)
                    if len(pose_history) > 0:
                        new_active_pose = Counter(pose_history).most_common(1)[0][0]
                        if new_active_pose != active_pose:
                            active_pose = new_active_pose
                            pose_name_ui, pose_hold_duration, pose_rest_duration, target_reps = get_pose_metadata(active_pose)
                            hold_timer = 0.0
                            rest_timer = 0.0
                            phase = "SETUP"

                checks, messages, spatial_ok, arms_down = evaluate_pose(
                    active_pose, rep_count, landmarks, world_landmarks, w,
                    hold_timer, rest_timer, in_hold_phase
                )

                # ── State Machine ──────────────────────────────────────────
                if phase == "SETUP":
                    if spatial_ok:
                        phase = "HOLD"
                        hold_timer = 0.0
                        play_phase_sound("HOLD")

                elif phase == "HOLD":
                    if spatial_ok:
                        hold_timer += dt
                        if hold_timer >= pose_hold_duration:
                            phase = "REST"
                            rest_timer = 0.0
                            play_phase_sound("REST")
                    else:
                        hold_timer = max(0.0, hold_timer - dt * 0.5)  # slow decay

                elif phase == "REST":
                    if arms_down:
                        rest_timer += dt
                        if rest_timer >= pose_rest_duration:
                            rep_count += 1
                            if rep_count >= target_reps:
                                phase = "COMPLETE"
                                play_phase_sound("COMPLETE")
                                hold_timer = 0.0
                            else:
                                phase = "SETUP"
                                hold_timer = 0.0
                                rest_timer = 0.0
                    else:
                        rest_timer = max(0.0, rest_timer - dt * 0.3)

                elif phase == "COMPLETE":
                    hold_timer += dt
                    if hold_timer > 2.0:
                        print("Exercise complete. Auto-exiting...")
                        break

                # Draw angle labels on elbows
                lms = landmarks
                l_angle = calculate_angle(
                    lms[LANDMARKS["LEFT_SHOULDER"]],
                    lms[LANDMARKS["LEFT_ELBOW"]],
                    lms[LANDMARKS["LEFT_WRIST"]]
                )
                r_angle = calculate_angle(
                    lms[LANDMARKS["RIGHT_SHOULDER"]],
                    lms[LANDMARKS["RIGHT_ELBOW"]],
                    lms[LANDMARKS["RIGHT_WRIST"]]
                )
                draw_angle_label(frame, lms[LANDMARKS["LEFT_ELBOW"]], l_angle, w, h)
                draw_angle_label(frame, lms[LANDMARKS["RIGHT_ELBOW"]], r_angle, w, h)

                # Coordinates panel
                draw_coordinates(frame, lms, w, h)

            else:
                # No person detected
                draw_feedback_banner(frame, "Stand in front of camera", COLOR_YELLOW)

            # ── UI ─────────────────────────────────────────────────────────
            # Status panel (right side)
            draw_status_panel(frame, checks, hold_timer, rest_timer, phase, fps, pose_name=active_pose)

            # Primary feedback banner (top left)
            banner_msg, banner_color = get_primary_feedback(checks, messages, phase)
            draw_feedback_banner(frame, banner_msg, banner_color)

            # Rep counter (and current target side for Warrior I / II / III)
            side_text = ""
            if active_pose in ["warrior1", "warrior2", "warrior3"]:
                target_side = "LEFT" if (rep_count % 2 == 0) else "RIGHT"
                side_text = f" | Next side: {target_side}"
                
            cv2.putText(frame, f"Reps: {rep_count}/{target_reps}{side_text}", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_YELLOW, 2)

            # Phase label
            cv2.putText(frame, f"[{phase}]", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

            # Pose Name label
            cv2.putText(frame, f"Target: {pose_name_ui}", (20, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)

            # Print coordinates to terminal
            if landmarks and int(fps_counter.fps) % 30 == 0:
                lms = landmarks
                print(f"[FPS:{fps:.1f}] Phase:{phase} | "
                      f"LWrist:({lms[15].x:.2f},{lms[15].y:.2f}) "
                      f"RWrist:({lms[16].x:.2f},{lms[16].y:.2f}) "
                      f"Hold:{hold_timer:.1f}s Rest:{rest_timer:.1f}s Reps:{rep_count}")

            cv2.imshow("Yoga Pose Correction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset
                hold_timer = 0.0
                rest_timer = 0.0
                rep_count = 0
                phase = "SETUP"
                if args.pose == "auto":
                    pose_history.clear()
                print("Reset!")

    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended. Reps completed: {rep_count}/{target_reps}")


if __name__ == "__main__":
    main()
