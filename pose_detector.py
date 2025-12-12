import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    def __init__(
        self,
        static_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(frame_rgb)

        # Extract landmarks
        landmarks = None
        if results.pose_landmarks:
            landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)

        return results, landmarks

    def _extract_landmarks(self, pose_landmarks, frame_shape):
        h, w, _ = frame_shape
        landmarks = {}

        # MediaPipe Pose landmark indices
        landmark_names = {
            0: "nose",
            1: "left_eye_inner",
            2: "left_eye",
            3: "left_eye_outer",
            4: "right_eye_inner",
            5: "right_eye",
            6: "right_eye_outer",
            7: "left_ear",
            8: "right_ear",
            9: "mouth_left",
            10: "mouth_right",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            17: "left_pinky",
            18: "right_pinky",
            19: "left_index",
            20: "right_index",
            21: "left_thumb",
            22: "right_thumb",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
            29: "left_heel",
            30: "right_heel",
            31: "left_foot_index",
            32: "right_foot_index",
        }

        for idx, landmark in enumerate(pose_landmarks.landmark):
            name = landmark_names.get(idx, f"landmark_{idx}")
            landmarks[name] = {
                "x": landmark.x * w,
                "y": landmark.y * h,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }

        return landmarks

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )
        return frame

    def close(self):
        self.pose.close()
