import numpy as np
from scipy.signal import savgol_filter


def calculate_angle(point1, point2, point3):
    # point2 is the vertex
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Calculate vectors
    vector1 = np.array([x1 - x2, y1 - y2])
    vector2 = np.array([x3 - x2, y3 - y2])

    # Calculate angle using dot product
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to avoid numerical errors
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_vertical_alignment(point1, point2, threshold=20):
    x1, _ = point1
    x2, _ = point2
    return abs(x2 - x1) < threshold


def calculate_horizontal_alignment(point1, point2, threshold=20):
    _, y1 = point1
    _, y2 = point2
    return abs(y2 - y1) < threshold


def calculate_symmetry(left_point, right_point, center_point):
    left_dist = calculate_distance(left_point, center_point)
    right_dist = calculate_distance(right_point, center_point)

    if left_dist + right_dist == 0:
        return 0

    symmetry = abs(left_dist - right_dist) / (left_dist + right_dist)
    return symmetry


def smooth_time_series(data, window_length=5, polyorder=2):
    if len(data) < window_length:
        return np.array(data)

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    return savgol_filter(data, window_length, polyorder)


def get_landmark_coords(landmarks, name):
    if landmarks is None or name not in landmarks:
        return None

    landmark = landmarks[name]
    return (landmark["x"], landmark["y"])


def check_visibility(landmarks, names, threshold=0.5):
    if landmarks is None:
        return False

    for name in names:
        if name not in landmarks or landmarks[name]["visibility"] < threshold:
            return False

    return True
