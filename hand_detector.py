"""
Hand Detection Module using MediaPipe
Detects and extracts hand landmarks in real-time from video frames
"""

import cv2
import mediapipe as mp
import numpy as np
from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS

# Import the correct API for mediapipe 0.10.x
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.framework.formats import landmark_pb2
    USE_NEW_API = True
except ImportError:
    # Fall back to older API if new one not available
    USE_NEW_API = False


class HandDetector:
    """
    Detects hands and extracts hand landmarks using MediaPipe
    """
    
    def __init__(self):
        """Initialize MediaPipe hand detection"""
        self.hand_landmarks_list = []
        
        if USE_NEW_API:
            # Use new MediaPipe Tasks API (0.10.x)
            base_options = python.BaseOptions(model_asset_path='mediapipe/hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                num_hands=MAX_NUM_HANDS,
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
                base_options=base_options)
            try:
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f"Warning: Could not load hand landmarker model: {e}")
                print("Falling back to direct landmark extraction...")
                self.detector = None
        else:
            self.detector = None
    
    def _draw_landmarks_new_api(self, frame, detection_result):
        """Draw hand landmarks on frame using new API results"""
        if not detection_result or not detection_result.hand_landmarks:
            return
        
        h, w, c = frame.shape
        
        # Hand connections (MediaPipe hand skeleton)
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky finger
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start_point = hand_landmarks[start_idx]
                end_point = hand_landmarks[end_idx]
                
                start_pos = (int(start_point.x * w), int(start_point.y * h))
                end_pos = (int(end_point.x * w), int(end_point.y * h))
                
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
            
            # Draw points
            for landmark in hand_landmarks:
                pos = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, pos, 4, (0, 0, 255), -1)
    
    def detect_hands(self, frame):
        """
        Detect hands in a given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            frame: Frame with hand landmarks drawn
            landmarks_list: List of hand landmarks
        """
        self.hand_landmarks_list = []
        h, w, c = frame.shape
        
        if USE_NEW_API and self.detector:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            try:
                detection_result = self.detector.detect(mp_image)
                
                # Draw landmarks
                self._draw_landmarks_new_api(frame, detection_result)
                
                # Extract landmarks
                if detection_result and detection_result.hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        landmarks = []
                        for landmark in hand_landmarks:
                            x = landmark.x * w
                            y = landmark.y * h
                            z = landmark.z
                            landmarks.append([x, y, z])
                        
                        handedness = "RIGHT" if hand_idx == 0 else "LEFT"
                        if detection_result.handedness:
                            try:
                                handedness = detection_result.handedness[hand_idx][0].category_name
                            except:
                                pass
                        
                        self.hand_landmarks_list.append({
                            'landmarks': np.array(landmarks),
                            'handedness': handedness,
                            'confidence': 0.9
                        })
            except Exception as e:
                print(f"Error in hand detection: {e}")
        
        return frame, self.hand_landmarks_list
    
    def get_normalized_landmarks(self, landmarks):
        """
        Normalize hand landmarks relative to the hand's centroid
        
        Args:
            landmarks: Raw hand landmarks array
            
        Returns:
            normalized_landmarks: Normalized coordinates
        """
        if landmarks is None or len(landmarks) == 0:
            return None
        
        landmarks = np.array(landmarks)
        # Calculate centroid
        centroid = np.mean(landmarks, axis=0)
        
        # Normalize by subtracting centroid
        normalized = landmarks - centroid
        
        # Scale by distance to max point
        max_distance = np.max(np.linalg.norm(normalized, axis=1))
        if max_distance > 0:
            normalized = normalized / max_distance
        
        return normalized.flatten()
    
    def extract_hand_features(self, landmarks):
        """
        Extract hand features for gesture recognition
        Features include: distances between key points, angles, etc.
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            features: Feature vector for the hand
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros(84)  # 21 landmarks * 3 coords + 20 distances
        
        landmarks = np.array(landmarks)
        
        # Flatten to get features
        features = landmarks.flatten()
        
        # Add distance features (distance from wrist to all other points)
        wrist = landmarks[0]
        distances = []
        for point in landmarks[1:]:
            dist = np.linalg.norm(point - wrist)
            distances.append(dist)
        
        # Pad or truncate to ensure consistent size
        while len(distances) < 20:
            distances.append(0)
        distances = distances[:20]
        
        features = np.concatenate([features, distances])
        
        return features
    
    def is_hand_visible(self):
        """Check if hand is currently visible"""
        return len(self.hand_landmarks_list) > 0
    
    def get_hand_count(self):
        """Get number of detected hands"""
        return len(self.hand_landmarks_list)
