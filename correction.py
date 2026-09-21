# correction.py — Rule-based pose correction engine
# All 6 checks for Urdhva Hastasana (Raised Hands Pose)

from config import *
from utils import (calculate_angle, landmark_visible,
                   pixel_distance, world_distance_cm)


def check_feet_apart(landmarks, world_landmarks):
    """
    Check 1: Feet ~15 cm apart.

    Strategy: Use ankle/hip pixel ratio as a camera-distance-independent measure.
    The average adult hip width is ~35 cm. The exercise requires 15 cm foot gap.
    So the ankle separation should be roughly 0.3–0.7x the hip width.

    We also try world landmarks as a secondary check, but pixel ratio is primary
    because world landmark depth estimation is unreliable at typical webcam distances.
    """
    lms = landmarks

    la = lms[LANDMARKS["LEFT_ANKLE"]]
    ra = lms[LANDMARKS["RIGHT_ANKLE"]]
    lh = lms[LANDMARKS["LEFT_HIP"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]

    if not all(landmark_visible(x) for x in [la, ra, lh, rh]):
        return False, "Ankles/hips not visible"

    # Pixel-based ratio: ankle_gap / hip_width
    ankle_gap = abs(la.x - ra.x)
    hip_width  = abs(lh.x - rh.x)

    if hip_width < 0.01:
        return False, "Cannot measure — stand facing camera"

    ratio = ankle_gap / hip_width

    # DEBUG — print live ratio so we can tune the threshold
    print(f"[FEET DEBUG] ankle_gap={ankle_gap:.3f}  hip_width={hip_width:.3f}  ratio={ratio:.3f}")

    # Very loose thresholds — adjust after seeing debug output
    RATIO_MIN = 1.0
    RATIO_MAX = 2.0

    ok = RATIO_MIN <= ratio <= RATIO_MAX

    # Also try world landmarks for display info
    world_str = ""
    if world_landmarks:
        try:
            wla = world_landmarks[LANDMARKS["LEFT_ANKLE"]]
            wra = world_landmarks[LANDMARKS["RIGHT_ANKLE"]]
            world_dist = world_distance_cm(wla, wra)
            world_str = f" (~{world_dist:.0f}cm)"
        except Exception:
            pass

    if ratio < RATIO_MIN:
        msg = f"Feet too close{world_str} — spread apart slightly"
    elif ratio > RATIO_MAX:
        msg = f"Feet too wide{world_str} — bring feet closer"
    else:
        msg = f"Feet OK{world_str} (ratio:{ratio:.2f})"

    return ok, msg


def check_arms_stretched(landmarks):
    """
    Check 2: Elbow angle > 160° on both sides (arms fully extended).
    Threshold: 160° — physiotherapy definition of functional full extension.
    """
    lms = landmarks

    ls = lms[LANDMARKS["LEFT_SHOULDER"]]
    le = lms[LANDMARKS["LEFT_ELBOW"]]
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rs = lms[LANDMARKS["RIGHT_SHOULDER"]]
    re = lms[LANDMARKS["RIGHT_ELBOW"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    # Visibility filter
    if not all(landmark_visible(x) for x in [ls, le, lw, rs, re, rw]):
        return False, "Move into frame — arms not fully visible"

    l_angle = calculate_angle(ls, le, lw)
    r_angle = calculate_angle(rs, re, rw)

    if l_angle is None or r_angle is None:
        return False, "Cannot compute arm angle"

    l_ok = l_angle > ELBOW_ANGLE_MIN
    r_ok = r_angle > ELBOW_ANGLE_MIN

    if l_ok and r_ok:
        return True, f"Arms straight (L:{int(l_angle)}° R:{int(r_angle)}°)"
    elif not l_ok and not r_ok:
        return False, f"Straighten both arms (L:{int(l_angle)}° R:{int(r_angle)}°)"
    elif not l_ok:
        return False, f"Straighten LEFT arm ({int(l_angle)}°)"
    else:
        return False, f"Straighten RIGHT arm ({int(r_angle)}°)"


def check_arms_near_ears(landmarks, frame_w):
    """
    Check 3: Wrists horizontally aligned with ears.
    Threshold: horizontal distance < 12% of frame width.
    """
    lms = landmarks
    le = lms[LANDMARKS["LEFT_EAR"]]
    re = lms[LANDMARKS["RIGHT_EAR"]]
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    if not all(landmark_visible(x) for x in [le, re, lw, rw]):
        return False, "Ears or wrists not visible"

    l_dist = abs(lw.x - le.x) * frame_w
    r_dist = abs(rw.x - re.x) * frame_w
    threshold = ARM_EAR_THRESHOLD * frame_w

    l_ok = l_dist < threshold
    r_ok = r_dist < threshold

    if l_ok and r_ok:
        return True, "Arms close to ears ✓"
    elif not l_ok and not r_ok:
        return False, "Bring BOTH arms closer to ears"
    elif not l_ok:
        return False, "Bring LEFT arm closer to ear"
    else:
        return False, "Bring RIGHT arm closer to ear"


def check_palms_joined(landmarks, frame_w):
    """
    Check 4: Both wrists close together (palms joined).
    Threshold: wrist distance < 6% of frame width.
    """
    lms = landmarks
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    if not (landmark_visible(lw) and landmark_visible(rw)):
        return False, "Wrists not visible"

    dist = abs(lw.x - rw.x) * frame_w
    threshold = PALMS_JOINED_THRESHOLD * frame_w

    ok = dist < threshold
    msg = "Palms joined ✓" if ok else f"Join your palms (gap: {int(dist)}px)"
    return ok, msg


def check_hold_timer(hold_timer, hold_duration=HOLD_DURATION, hold_breaths=HOLD_BREATHS):
    """
    Check 5: Posture held for required complete breaths.
    Timer managed externally; this just evaluates completion.
    """
    ok = hold_timer >= hold_duration
    breaths = min(int(hold_timer / BREATH_DURATION), hold_breaths)
    msg = (f"Hold complete! ({hold_breaths} breaths)" if ok
           else f"Hold... {breaths}/{hold_breaths} breaths ({hold_timer:.1f}s)")
    return ok, msg


def check_rest_phase(landmarks, rest_timer, rest_duration=REST_DURATION, rest_breaths=REST_BREATHS):
    """
    Check 6: Arms lowered to thighs for rest.
    y increases downward in image coords — wrist.y >= hip.y means arms are down.
    """
    lms = landmarks
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]
    lh = lms[LANDMARKS["LEFT_HIP"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]

    if not all(landmark_visible(x) for x in [lw, rw, lh, rh]):
        return False, False, "Hips/wrists not visible"

    arms_down = (lw.y >= lh.y - REST_TOLERANCE and
                 rw.y >= rh.y - REST_TOLERANCE)

    rest_complete = rest_timer >= rest_duration
    breaths = min(int(rest_timer / BREATH_DURATION), rest_breaths)

    if rest_complete:
        msg = "Rest complete! Ready for next rep."
    elif arms_down:
        msg = f"Resting... {breaths}/{rest_breaths} breaths ({rest_timer:.1f}s)"
    else:
        msg = "Lower arms to thighs to rest"

    return arms_down, rest_complete, msg


def evaluate_urdhva_hastasana(landmarks, world_landmarks, frame_w,
                              hold_timer, rest_timer, in_hold_phase):
    """
    Run all 6 checks for Urdhva Hastasana and return:
      - checks: list of 6 booleans
      - messages: list of 6 feedback strings
      - overall_ok: True if checks 1-4 all pass (spatial checks)
    """
    c1, m1 = check_feet_apart(landmarks, world_landmarks)
    c2, m2 = check_arms_stretched(landmarks)
    c3, m3 = check_arms_near_ears(landmarks, frame_w)
    c4, m4 = check_palms_joined(landmarks, frame_w)
    c5, m5 = check_hold_timer(hold_timer)

    arms_down, rest_complete, m6 = check_rest_phase(landmarks, rest_timer)
    c6 = rest_complete

    checks = [c1, c2, c3, c4, c5, c6]
    messages = [m1, m2, m3, m4, m5, m6]
    spatial_ok = c1 and c2 and c3 and c4   # all position checks pass

    return checks, messages, spatial_ok, arms_down


# ── Warrior II Pose Checks ──────────────────────────────────────────────────

def check_warrior_stance(landmarks):
    """Check 1: Feet wide apart."""
    lms = landmarks
    la = lms[LANDMARKS["LEFT_ANKLE"]]
    ra = lms[LANDMARKS["RIGHT_ANKLE"]]
    lh = lms[LANDMARKS["LEFT_HIP"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]

    if not all(landmark_visible(x) for x in [la, ra, lh, rh]):
        return False, "Ankles/hips not visible"

    ankle_gap = abs(la.x - ra.x)
    hip_width = abs(lh.x - rh.x)

    if hip_width < 0.01:
        return False, "Cannot measure — stand facing camera"

    ratio = ankle_gap / hip_width
    ok = ratio >= WARRIOR_FEET_DIST_MIN_RATIO

    if ok:
        msg = f"Stance wide enough ✓ (ratio:{ratio:.1f})"
    else:
        msg = f"Widen your stance (ratio:{ratio:.1f})"

    return ok, msg


def check_warrior_arms(landmarks):
    """Check 2: Arms stretched horizontally."""
    lms = landmarks
    ls = lms[LANDMARKS["LEFT_SHOULDER"]]
    le = lms[LANDMARKS["LEFT_ELBOW"]]
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rs = lms[LANDMARKS["RIGHT_SHOULDER"]]
    re = lms[LANDMARKS["RIGHT_ELBOW"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    if not all(landmark_visible(x) for x in [ls, le, lw, rs, re, rw]):
        return False, "Arms not fully visible"

    l_angle = calculate_angle(ls, le, lw)
    r_angle = calculate_angle(rs, re, rw)

    if l_angle is None or r_angle is None:
        return False, "Cannot compute arm angle"

    l_ok = l_angle > WARRIOR_ARM_STRAIGHT_MIN
    r_ok = r_angle > WARRIOR_ARM_STRAIGHT_MIN

    if l_ok and r_ok:
        return True, "Arms extended ✓"
    elif not l_ok and not r_ok:
        return False, "Extend both arms straight"
    elif not l_ok:
        return False, "Extend LEFT arm"
    else:
        return False, "Extend RIGHT arm"


def check_warrior_legs(landmarks, rep_count):
    """
    Check 3 & 4: Front knee bent, back leg straight.
    Target side alternates based on rep_count.
    Even rep (0, 2, 4...) -> Left leg bent.
    Odd rep (1, 3, 5...) -> Right leg bent.
    """
    lms = landmarks
    lh = lms[LANDMARKS["LEFT_HIP"]]
    lk = lms[LANDMARKS["LEFT_KNEE"]]
    la = lms[LANDMARKS["LEFT_ANKLE"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]
    rk = lms[LANDMARKS["RIGHT_KNEE"]]
    ra = lms[LANDMARKS["RIGHT_ANKLE"]]

    if not all(landmark_visible(x) for x in [lh, lk, la, rh, rk, ra]):
        return False, "Legs not visible", False, "Legs not visible"

    l_angle = calculate_angle(lh, lk, la)
    r_angle = calculate_angle(rh, rk, ra)

    if l_angle is None or r_angle is None:
        return False, "Cannot compute leg angles", False, "Cannot compute leg angles"

    target_is_left = (rep_count % 2 == 0)
    target_side_str = "LEFT" if target_is_left else "RIGHT"

    if target_is_left:
        front_angle = l_angle
        back_angle = r_angle
    else:
        front_angle = r_angle
        back_angle = l_angle

    # Check 3: Front knee bent
    front_ok = WARRIOR_KNEE_BENT_MIN <= front_angle <= WARRIOR_KNEE_BENT_MAX
    if front_ok:
        msg3 = f"{target_side_str} knee bent ✓ ({int(front_angle)}°)"
    elif front_angle > WARRIOR_KNEE_BENT_MAX:
        msg3 = f"Bend {target_side_str} knee more ({int(front_angle)}°)"
    else:
        msg3 = f"Straighten {target_side_str} knee a bit ({int(front_angle)}°)"

    # Check 4: Back leg straight
    back_ok = back_angle > WARRIOR_KNEE_STRAIGHT_MIN
    back_side_str = "RIGHT" if target_is_left else "LEFT"
    if back_ok:
        msg4 = f"{back_side_str} leg straight ✓ ({int(back_angle)}°)"
    else:
        msg4 = f"Straighten {back_side_str} leg ({int(back_angle)}°)"

    return front_ok, msg3, back_ok, msg4


def evaluate_warrior_ii(rep_count, landmarks, world_landmarks, frame_w,
                        hold_timer, rest_timer, in_hold_phase):
    """Run checks for Warrior II."""
    c1, m1 = check_warrior_stance(landmarks)
    c2, m2 = check_warrior_arms(landmarks)
    
    # c3=front leg, c4=back leg
    c3, m3, c4, m4 = check_warrior_legs(landmarks, rep_count)
    
    c5, m5 = check_hold_timer(hold_timer, WARRIOR_HOLD_DURATION, WARRIOR_HOLD_BREATHS)

    # For rest phase, we can just use the arms down check like Urdhva, or a simple standing pose.
    # Let's reuse check_rest_phase (arms to side) for rest.
    arms_down, rest_complete, m6 = check_rest_phase(landmarks, rest_timer, WARRIOR_REST_DURATION, WARRIOR_REST_BREATHS)
    c6 = rest_complete

    checks = [c1, c2, c3, c4, c5, c6]
    messages = [m1, m2, m3, m4, m5, m6]
    spatial_ok = c1 and c2 and c3 and c4

    return checks, messages, spatial_ok, arms_down


# ── Warrior I Pose Checks ───────────────────────────────────────────────────

def check_warrior1_arms(landmarks, frame_w):
    """Check 2 (Warrior I): Arms stretched AND palms joined."""
    c1, m1 = check_arms_stretched(landmarks)
    c2, m2 = check_palms_joined(landmarks, frame_w)

    ok = c1 and c2
    if not c1:
        msg = m1
    elif not c2:
        msg = m2
    else:
        msg = "Arms raised & palms joined ✓"
    
    return ok, msg


def evaluate_warrior_i(rep_count, landmarks, world_landmarks, frame_w,
                       hold_timer, rest_timer, in_hold_phase):
    """Run checks for Warrior I."""
    c1, m1 = check_warrior_stance(landmarks)
    c2, m2 = check_warrior1_arms(landmarks, frame_w)
    
    # c3=front leg, c4=back leg
    c3, m3, c4, m4 = check_warrior_legs(landmarks, rep_count)
    
    c5, m5 = check_hold_timer(hold_timer, WARRIOR1_HOLD_DURATION, WARRIOR1_HOLD_BREATHS)
    
    arms_down, rest_complete, m6 = check_rest_phase(landmarks, rest_timer, WARRIOR1_REST_DURATION, WARRIOR1_REST_BREATHS)
    c6 = rest_complete

    checks = [c1, c2, c3, c4, c5, c6]
    messages = [m1, m2, m3, m4, m5, m6]
    spatial_ok = c1 and c2 and c3 and c4

    return checks, messages, spatial_ok, arms_down


# ── Warrior III Pose Checks ─────────────────────────────────────────────────

def check_warrior3_legs(landmarks, rep_count):
    """
    Check 1 & 2: Standing leg straight, Raised leg straight.
    Target side alternates based on rep_count.
    Even rep (0, 2, 4...) -> Left leg standing, Right leg raised.
    Odd rep (1, 3, 5...) -> Right leg standing, Left leg raised.
    """
    lms = landmarks
    lh = lms[LANDMARKS["LEFT_HIP"]]
    lk = lms[LANDMARKS["LEFT_KNEE"]]
    la = lms[LANDMARKS["LEFT_ANKLE"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]
    rk = lms[LANDMARKS["RIGHT_KNEE"]]
    ra = lms[LANDMARKS["RIGHT_ANKLE"]]

    if not all(landmark_visible(x) for x in [lh, lk, la, rh, rk, ra]):
        return False, "Legs not visible", False, "Legs not visible"

    l_angle = calculate_angle(lh, lk, la)
    r_angle = calculate_angle(rh, rk, ra)

    if l_angle is None or r_angle is None:
        return False, "Cannot compute leg angles", False, "Cannot compute leg angles"

    standing_is_left = (rep_count % 2 == 0)
    
    standing_angle = l_angle if standing_is_left else r_angle
    raised_angle = r_angle if standing_is_left else l_angle
    
    standing_side = "LEFT" if standing_is_left else "RIGHT"
    raised_side = "RIGHT" if standing_is_left else "LEFT"

    standing_ok = standing_angle > WARRIOR3_KNEE_STRAIGHT_MIN
    if standing_ok:
        m1 = f"Standing leg straight ✓"
    else:
        m1 = f"Straighten standing leg ({int(standing_angle)}°)"

    raised_ok = raised_angle > WARRIOR3_KNEE_STRAIGHT_MIN
    if raised_ok:
        m2 = f"Raised leg straight ✓"
    else:
        m2 = f"Straighten raised leg ({int(raised_angle)}°)"

    return standing_ok, m1, raised_ok, m2


def check_warrior3_alignment(landmarks, rep_count):
    """
    Check 3 & 4: Torso/Leg parallel (Shoulder-Hip-RaisedAnkle),
                 Arms forward (Wrist-Shoulder-Hip).
    """
    lms = landmarks
    ls = lms[LANDMARKS["LEFT_SHOULDER"]]
    rs = lms[LANDMARKS["RIGHT_SHOULDER"]]
    lh = lms[LANDMARKS["LEFT_HIP"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]
    la = lms[LANDMARKS["LEFT_ANKLE"]]
    ra = lms[LANDMARKS["RIGHT_ANKLE"]]
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    if not all(landmark_visible(x) for x in [ls, rs, lh, rh, la, ra, lw, rw]):
        return False, "Body not fully visible", False, "Body not fully visible"

    standing_is_left = (rep_count % 2 == 0)
    
    # We use the raised side's shoulder, hip, and ankle for torso/leg alignment
    # Wait, usually torso is twisted slightly or squared. We can average or just use one side.
    # Let's use the standing side's shoulder/hip for better visibility since it's the pivot.
    shoulder = ls if standing_is_left else rs
    hip = lh if standing_is_left else rh
    raised_ankle = ra if standing_is_left else la
    wrist = lw if standing_is_left else rw

    # Check 3: Torso-Leg straightness & Waist bend
    body_angle = calculate_angle(shoulder, hip, raised_ankle)
    standing_knee = lms[LANDMARKS["LEFT_KNEE"]] if standing_is_left else lms[LANDMARKS["RIGHT_KNEE"]]
    bend_angle = calculate_angle(standing_knee, hip, shoulder)

    if body_angle is None or bend_angle is None:
        return False, "Cannot compute body alignment", False, "Cannot compute arm alignment"
    
    is_straight = body_angle > WARRIOR3_BODY_ALIGN_MIN
    is_bent_over = bend_angle < WARRIOR3_WAIST_BEND_MAX  # If upright, it's ~180. If horizontal 'T', it's ~90.

    body_ok = is_straight and is_bent_over

    if body_ok:
        m3 = "Torso & leg aligned ✓"
    elif not is_bent_over:
        m3 = f"Bend forward from waist ({int(bend_angle)}°)"
    else:
        m3 = f"Straighten body line ({int(body_angle)}°)"

    # Check 4: Arm-Torso straightness
    arm_angle = calculate_angle(wrist, shoulder, hip)
    if arm_angle is None:
        m4 = "Cannot compute arm alignment"
        arm_ok = False
    else:
        arm_ok = arm_angle > WARRIOR3_ARM_ALIGN_MIN
        m4 = "Arms extended forward ✓" if arm_ok else f"Raise arms in line with body ({int(arm_angle)}°)"

    return body_ok, m3, arm_ok, m4


def evaluate_warrior_iii(rep_count, landmarks, world_landmarks, frame_w,
                         hold_timer, rest_timer, in_hold_phase):
    """Run checks for Warrior III."""
    c1, m1, c2, m2 = check_warrior3_legs(landmarks, rep_count)
    c3, m3, c4, m4 = check_warrior3_alignment(landmarks, rep_count)
    
    c5, m5 = check_hold_timer(hold_timer, WARRIOR3_HOLD_DURATION, WARRIOR3_HOLD_BREATHS)
    
    arms_down, rest_complete, m6 = check_rest_phase(landmarks, rest_timer, WARRIOR3_REST_DURATION, WARRIOR3_REST_BREATHS)
    c6 = rest_complete

    checks = [c1, c2, c3, c4, c5, c6]
    messages = [m1, m2, m3, m4, m5, m6]
    spatial_ok = c1 and c2 and c3 and c4

    return checks, messages, spatial_ok, arms_down


# ── Router ──────────────────────────────────────────────────────────────────

def evaluate_pose(pose_name, rep_count, landmarks, world_landmarks, frame_w,
                  hold_timer, rest_timer, in_hold_phase):
    if pose_name == "warrior2":
        return evaluate_warrior_ii(rep_count, landmarks, world_landmarks, frame_w,
                                   hold_timer, rest_timer, in_hold_phase)
    elif pose_name == "warrior1":
        return evaluate_warrior_i(rep_count, landmarks, world_landmarks, frame_w,
                                  hold_timer, rest_timer, in_hold_phase)
    elif pose_name == "warrior3":
        return evaluate_warrior_iii(rep_count, landmarks, world_landmarks, frame_w,
                                    hold_timer, rest_timer, in_hold_phase)
    else:
        return evaluate_urdhva_hastasana(landmarks, world_landmarks, frame_w,
                                         hold_timer, rest_timer, in_hold_phase)

def autodetect_pose(landmarks):
    """
    Heuristic to guess which of the 4 poses the user is attempting.
    Returns: "urdhva", "warrior1", "warrior2", or "warrior3"
    """
    lms = landmarks
    ls = lms[LANDMARKS["LEFT_SHOULDER"]]
    rs = lms[LANDMARKS["RIGHT_SHOULDER"]]
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]
    la = lms[LANDMARKS["LEFT_ANKLE"]]
    ra = lms[LANDMARKS["RIGHT_ANKLE"]]
    lh = lms[LANDMARKS["LEFT_HIP"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]

    if not all(landmark_visible(x) for x in [ls, rs, lw, rw, la, ra, lh, rh]):
        return "urdhva"

    hip_width = max(abs(lh.x - rh.x), 0.01)
    ankle_gap = abs(la.x - ra.x)
    
    stance_wide = (ankle_gap / hip_width) > 1.5

    arms_raised = lw.y < ls.y - 0.1 and rw.y < rs.y - 0.1
    arms_horizontal = abs(lw.y - ls.y) < 0.15 and abs(rw.y - rs.y) < 0.15

    torso_horizontal = abs(ls.y - lh.y) < 0.2 and abs(rs.y - rh.y) < 0.2
    leg_raised = abs(la.y - lh.y) < 0.3 or abs(ra.y - rh.y) < 0.3

    if torso_horizontal and leg_raised:
        return "warrior3"
    
    if arms_horizontal and stance_wide:
        return "warrior2"
        
    if arms_raised and stance_wide:
        return "warrior1"
        
    return "urdhva"
