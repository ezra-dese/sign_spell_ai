import numpy as np
import math

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, landmarks):
        """
        Convert 21 landmarks into a feature vector.
        Features:
        - Angles of 5 fingers (0-180 degrees)
        - Distances between tips (optional, normalized)
        """
        if not landmarks or len(landmarks) != 21:
            return []

        # Convert to numpy array for easier math
        # landmarks is list of [id, x, y]
        # We only need x, y
        points = np.array([[lm[1], lm[2]] for lm in landmarks])

        # 1. Normalize coordinates
        # Shift so wrist (0) is at (0,0)
        wrist = points[0]
        points = points - wrist

        # Scale by hand size (distance between wrist and middle finger MCP (9))
        # This makes it invariant to camera distance
        hand_size = np.linalg.norm(points[9])
        if hand_size > 0:
            points = points / hand_size

        features = []

        # 2. Calculate Angles
        # Thumb: 2-3-4
        features.append(self.get_angle(points[2], points[3], points[4]))
        # Index: 5-6-7-8 (Use 5-6-8 for general bend?) 
        # Better: Angle at joint 6 (5-6-7) and joint 7 (6-7-8)
        # For simplicity, let's use the angle of the finger relative to the palm or just the bend angle.
        
        # Let's use 3 angles per finger for detailed shape, or just one "bend" angle.
        # For sign language, we need to know if a finger is straight or curled.
        
        # Thumb
        features.append(self.get_angle(points[0], points[2], points[4]))

        # Fingers (Index, Middle, Ring, Pinky)
        # Tips: 8, 12, 16, 20
        # PIP joints: 6, 10, 14, 18
        # MCP joints: 5, 9, 13, 17
        
        # Index
        features.append(self.get_angle(points[0], points[5], points[8])) # Openness
        features.append(self.get_angle(points[5], points[6], points[7])) # Joint bend

        # Middle
        features.append(self.get_angle(points[0], points[9], points[12]))
        features.append(self.get_angle(points[9], points[10], points[11]))

        # Ring
        features.append(self.get_angle(points[0], points[13], points[16]))
        features.append(self.get_angle(points[13], points[14], points[15]))

        # Pinky
        features.append(self.get_angle(points[0], points[17], points[20]))
        features.append(self.get_angle(points[17], points[18], points[19]))

        # 3. Distances between tips (useful for 'O' vs 'C' etc)
        # Thumb tip to Index tip
        features.append(np.linalg.norm(points[4] - points[8]))
        
        return features

    def get_angle(self, a, b, c):
        """Calculate angle at b given points a, b, c"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
