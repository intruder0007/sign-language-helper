"""
Data Collector Module for Training Data
Collects hand landmark data for specific gestures to train the model
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from hand_detector import HandDetector
import config


class GestureDataCollector:
    """
    Collects training data for gesture recognition
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.hand_detector = HandDetector()
        self.cap = None
        self.gesture_name = None
        self.collected_samples = 0
        self.max_samples = 100
    
    def setup_camera(self):
        """Setup camera capture"""
        self.cap = cv2.VideoCapture(config.CAMERA_ID)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        return True
    
    def create_gesture_folder(self, gesture_name):
        """
        Create folder for gesture data
        
        Args:
            gesture_name: Name of the gesture
        """
        gesture_dir = self.data_dir / gesture_name
        gesture_dir.mkdir(exist_ok=True)
        return gesture_dir
    
    def save_gesture_data(self, features, gesture_name):
        """
        Save gesture feature vector
        
        Args:
            features: Feature vector
            gesture_name: Name of gesture
        """
        gesture_dir = self.create_gesture_folder(gesture_name)
        
        # Generate filename based on current count
        data_file = gesture_dir / f"{gesture_name}_{self.collected_samples:03d}.npy"
        np.save(str(data_file), features)
        
        self.collected_samples += 1
    
    def collect_data_for_gesture(self, gesture_name, num_samples=100):
        """
        Collect training data for a specific gesture
        
        Args:
            gesture_name: Name of the gesture to collect data for
            num_samples: Number of samples to collect
        """
        if not self.setup_camera():
            return
        
        self.gesture_name = gesture_name
        self.collected_samples = 0
        self.max_samples = num_samples
        
        print(f"\nStarting data collection for gesture: {gesture_name}")
        print(f"Collecting {num_samples} samples...")
        print("Position your hand in the frame and press SPACE to capture")
        print("Press 'q' to quit collection")
        
        while self.collected_samples < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            frame, hand_landmarks_list = self.hand_detector.detect_hands(frame)
            
            # Display instruction
            text = f"Collecting '{gesture_name}': {self.collected_samples}/{num_samples}"
            cv2.putText(
                frame, text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, "Press SPACE to capture | 'q' to quit", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
            )
            
            cv2.imshow(f'Data Collection - {gesture_name}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and hand_landmarks_list:
                # Capture hand data
                for hand_data in hand_landmarks_list:
                    landmarks = hand_data['landmarks']
                    features = self.hand_detector.extract_hand_features(landmarks)
                    self.save_gesture_data(features, gesture_name)
                    print(f"Captured sample {self.collected_samples}/{num_samples}")
            
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"Data collection completed. Collected {self.collected_samples} samples for '{gesture_name}'")
    
    def load_all_data(self):
        """
        Load all collected gesture data
        
        Returns:
            X: Feature vectors
            y: Labels (gesture names)
        """
        X = []
        y = []
        
        for gesture_folder in self.data_dir.iterdir():
            if gesture_folder.is_dir():
                gesture_name = gesture_folder.name
                
                for data_file in gesture_folder.glob("*.npy"):
                    features = np.load(str(data_file))
                    X.append(features)
                    y.append(gesture_name)
        
        if X:
            X = np.array(X)
            y = np.array(y)
            return X, y
        
        return None, None


def interactive_collection():
    """Interactive mode for collecting gesture data"""
    collector = GestureDataCollector()
    
    print("\n=== Sign Language Gesture Data Collector ===")
    print("\nAvailable gestures to collect data for:")
    
    gestures = list(config.SUPPORTED_GESTURES.keys())
    for i, gesture in enumerate(gestures):
        print(f"  {i+1}. {gesture}")
    
    while True:
        choice = input("\nEnter gesture name or number (or 'q' to quit): ").strip().upper()
        
        if choice == 'Q':
            break
        
        if choice.isdigit() and 1 <= int(choice) <= len(gestures):
            gesture_name = gestures[int(choice) - 1]
        else:
            gesture_name = choice
        
        if gesture_name in config.SUPPORTED_GESTURES:
            samples = input(f"Number of samples to collect (default 100): ").strip()
            samples = int(samples) if samples.isdigit() else 100
            
            collector.collect_data_for_gesture(gesture_name, samples)
        else:
            print(f"Unknown gesture: {gesture_name}")


if __name__ == "__main__":
    interactive_collection()
