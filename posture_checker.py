from pose_utils import (
    calculate_angle,
    get_landmark_coords,
    check_visibility,
    calculate_symmetry,
    calculate_distance,
)


class PostureChecker:
    def __init__(self):
        self.feedback = []

    def check_bicep_curl(self, landmarks, side="left"):
        if side == "left":
            shoulder = "left_shoulder"
            elbow = "left_elbow"
            wrist = "left_wrist"
        else:
            shoulder = "right_shoulder"
            elbow = "right_elbow"
            wrist = "right_wrist"

        # Check if landmarks are visible
        required = [shoulder, elbow, wrist]
        if not check_visibility(landmarks, required):
            return {
                "valid": False,
                "feedback": [f"Cannot detect {side} arm landmarks clearly"],
                "metrics": {},
            }

        # Get coordinates
        shoulder_pos = get_landmark_coords(landmarks, shoulder)
        elbow_pos = get_landmark_coords(landmarks, elbow)
        wrist_pos = get_landmark_coords(landmarks, wrist)

        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder_pos, elbow_pos, wrist_pos)

        feedback = []
        valid = True

        # Rule 1: Check elbow angle range
        if elbow_angle < 30:
            feedback.append(f"{side.capitalize()} arm: Elbow too bent (curl too high)")
            valid = False
        elif elbow_angle > 160:
            feedback.append(
                f"{side.capitalize()} arm: Arm almost straight (lower position)"
            )

        # Rule 2: Check if wrist is above elbow (proper curl)
        if wrist_pos[1] > elbow_pos[1]:  # y increases downward
            feedback.append(f"{side.capitalize()} arm: Lift your wrist higher")
            valid = False

        # Rule 3: Check elbow position relative to shoulder (should be close)
        elbow_shoulder_dist = calculate_distance(elbow_pos, shoulder_pos)
        if elbow_shoulder_dist > 150:  # pixels, threshold for elbow swing
            feedback.append(
                f"{side.capitalize()} arm: Keep elbow closer to body (avoid swinging)"
            )
            valid = False

        if not feedback:
            feedback.append(f"{side.capitalize()} bicep curl: Good form!")

        return {
            "valid": valid,
            "feedback": feedback,
            "metrics": {
                "elbow_angle": elbow_angle,
                "elbow_shoulder_distance": elbow_shoulder_dist,
            },
        }

    def check_lateral_raise(self, landmarks, side="left"):
        if side == "left":
            shoulder = "left_shoulder"
            elbow = "left_elbow"
            wrist = "left_wrist"
        else:
            shoulder = "right_shoulder"
            elbow = "right_elbow"
            wrist = "right_wrist"

        # Check visibility
        required = [shoulder, elbow, wrist]
        if not check_visibility(landmarks, required):
            return {
                "valid": False,
                "feedback": [f"Cannot detect {side} arm landmarks clearly"],
                "metrics": {},
            }

        # Get coordinates
        shoulder_pos = get_landmark_coords(landmarks, shoulder)
        elbow_pos = get_landmark_coords(landmarks, elbow)
        wrist_pos = get_landmark_coords(landmarks, wrist)

        # Calculate arm angle (shoulder-elbow-wrist)
        arm_angle = calculate_angle(shoulder_pos, elbow_pos, wrist_pos)

        # Calculate vertical position difference
        wrist_shoulder_vertical = abs(wrist_pos[1] - shoulder_pos[1])

        feedback = []
        valid = True

        # Rule 1: Check if arm is raised to shoulder level
        if wrist_pos[1] > shoulder_pos[1] + 50:  # wrist significantly below shoulder
            feedback.append(
                f"{side.capitalize()} arm: Raise arm higher to shoulder level"
            )
            valid = False

        # Rule 2: Check if wrist is too high (above shoulder)
        if wrist_pos[1] < shoulder_pos[1] - 50:
            feedback.append(
                f"{side.capitalize()} arm: Don't raise wrist above shoulder level"
            )
            valid = False

        # Rule 3: Check elbow angle (should be slightly bent, not locked)
        if arm_angle < 140:
            feedback.append(f"{side.capitalize()} arm: Straighten your arm more")
            valid = False
        elif arm_angle > 175:
            feedback.append(
                f"{side.capitalize()} arm: Keep elbow slightly bent (don't lock)"
            )
            valid = False

        # Rule 4: Check horizontal alignment
        horizontal_dist = abs(wrist_pos[0] - shoulder_pos[0])
        if horizontal_dist < 100:  # too close horizontally
            feedback.append(f"{side.capitalize()} arm: Extend arm more to the side")
            valid = False

        if not feedback:
            feedback.append(f"{side.capitalize()} lateral raise: Good form!")

        return {
            "valid": valid,
            "feedback": feedback,
            "metrics": {
                "arm_angle": arm_angle,
                "wrist_shoulder_vertical_diff": wrist_shoulder_vertical,
                "horizontal_distance": horizontal_dist,
            },
        }

    def check_back_posture(self, landmarks):
        # Check visibility
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not check_visibility(landmarks, required):
            return {
                "valid": False,
                "feedback": ["Cannot detect body landmarks clearly"],
                "metrics": {},
            }

        # Get coordinates
        left_shoulder = get_landmark_coords(landmarks, "left_shoulder")
        right_shoulder = get_landmark_coords(landmarks, "right_shoulder")
        left_hip = get_landmark_coords(landmarks, "left_hip")
        right_hip = get_landmark_coords(landmarks, "right_hip")

        # Calculate midpoints
        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
        )
        hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        # Calculate angles and differences
        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
        hip_height_diff = abs(left_hip[1] - right_hip[1])

        # Calculate spine angle (should be close to vertical)
        spine_angle = calculate_angle(
            (shoulder_mid[0], shoulder_mid[1] - 100),  # point above shoulder
            shoulder_mid,
            hip_mid,
        )

        # Calculate symmetry
        shoulder_symmetry = calculate_symmetry(
            left_shoulder, right_shoulder, shoulder_mid
        )
        hip_symmetry = calculate_symmetry(left_hip, right_hip, hip_mid)

        feedback = []
        valid = True

        # Rule 1: Check shoulder level
        if shoulder_height_diff > 30:  # pixels
            feedback.append("Shoulders are not level - adjust posture")
            valid = False

        # Rule 2: Check hip level
        if hip_height_diff > 30:
            feedback.append("Hips are not level - balance your stance")
            valid = False

        # Rule 3: Check spine alignment (should be close to straight)
        if spine_angle < 160 or spine_angle > 200:
            feedback.append("Back is leaning - maintain straight posture")
            valid = False

        # Rule 4: Check overall symmetry
        if shoulder_symmetry > 0.2 or hip_symmetry > 0.2:
            feedback.append("Body is not balanced - center your weight")
            valid = False

        if not feedback:
            feedback.append("Back posture: Good form!")

        return {
            "valid": valid,
            "feedback": feedback,
            "metrics": {
                "shoulder_height_diff": shoulder_height_diff,
                "hip_height_diff": hip_height_diff,
                "spine_angle": spine_angle,
                "shoulder_symmetry": shoulder_symmetry,
                "hip_symmetry": hip_symmetry,
            },
        }

    def check_all(self, landmarks, exercise_type="bicep_curl", side="left"):
        results = {"exercise": exercise_type, "checks": {}}

        # Always check back posture
        results["checks"]["back_posture"] = self.check_back_posture(landmarks)

        # Exercise-specific checks
        if exercise_type == "bicep_curl":
            results["checks"]["bicep_curl"] = self.check_bicep_curl(landmarks, side)
        elif exercise_type == "lateral_raise":
            results["checks"]["lateral_raise"] = self.check_lateral_raise(
                landmarks, side
            )

        # Aggregate feedback
        all_feedback = []
        all_valid = True
        for check_name, check_result in results["checks"].items():
            all_feedback.extend(check_result["feedback"])
            if not check_result["valid"]:
                all_valid = False

        results["overall_valid"] = all_valid
        results["all_feedback"] = all_feedback

        return results
