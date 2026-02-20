"""
Sign Language Recognition App - Main Application
Real-time gesture recognition from video input
"""

import cv2
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer
import config


class SignLanguageRecognitionApp:
    """
    Main application for real-time sign language recognition
    """
    
    def __init__(self):
        """Initialize the application"""
        self.hand_detector = HandDetector()
        self.gesture_recognizer = GestureRecognizer()
        self.cap = None
        self.output_text = ""
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.frame_count = 0
        self.gesture_count = 0
        
    def setup_camera(self):
        """Setup camera capture"""
        self.cap = cv2.VideoCapture(config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        return True
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
    
    def display_info(self, frame):
        """Display information on the frame"""
        # Display output text
        cv2.putText(
            frame,
            f"Text: {self.output_text[-50:]}",  # Show last 50 characters
            config.OUTPUT_TEXT_POSITION,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            config.TEXT_COLOR,
            2
        )
        
        # Display current gesture
        if self.current_gesture:
            gesture_text = f"Gesture: {self.current_gesture} ({self.gesture_confidence:.2f})"
            cv2.putText(
                frame,
                gesture_text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                config.TEXT_COLOR,
                2
            )
        
        # Display instructions
        instructions = [
            "Press 'c' to clear text | 's' to save | 'q' to quit",
            "Move hands in frame to perform gestures"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                frame,
                text,
                (10, frame.shape[0] - 30 - (i * 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Display hand count
        hand_count = self.hand_detector.get_hand_count()
        cv2.putText(
            frame,
            f"Hands detected: {hand_count}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            config.TEXT_COLOR,
            2
        )
        
        # Display frame count
        cv2.putText(
            frame,
            f"Frame: {self.frame_count} | Gestures recognized: {self.gesture_count}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame for gesture recognition
        
        Args:
            frame: Input frame from camera
            
        Returns:
            frame: Processed frame with overlays
        """
        # Detect hands
        frame, hand_landmarks_list = self.hand_detector.detect_hands(frame)
        
        # Extract features if hands are detected
        if hand_landmarks_list:
            for hand_data in hand_landmarks_list:
                landmarks = hand_data['landmarks']
                features = self.hand_detector.extract_hand_features(landmarks)
                self.gesture_recognizer.add_frame_features(features)
            
            # Try to recognize gesture every few frames
            if self.frame_count % 10 == 0 and len(self.gesture_recognizer.gesture_sequence) >= config.GESTURE_SEQUENCE_LENGTH:
                gesture, confidence = self.gesture_recognizer.recognize_gesture()
                if gesture and confidence >= config.CONFIDENCE_THRESHOLD:
                    self.current_gesture = gesture
                    self.gesture_confidence = confidence
                    self.gesture_recognizer.add_to_history(gesture, confidence)
                    
                    # Update output text
                    if gesture in config.SUPPORTED_GESTURES:
                        char = config.SUPPORTED_GESTURES[gesture]
                        if gesture == 'DELETE':
                            self.output_text = self.output_text[:-1]
                        elif gesture != 'NOTHING':
                            self.output_text += char
                            self.gesture_count += 1
                    
                    # Reset for next gesture
                    self.gesture_recognizer.reset_sequence()
        else:
            # No hands detected, reset sequences
            self.gesture_recognizer.reset_sequence()
            self.current_gesture = None
        
        # Display information
        frame = self.display_info(frame)
        
        return frame
    
    def handle_input(self, key):
        """
        Handle keyboard input
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            bool: False if quit was requested
        """
        if key == ord('q'):
            return False
        elif key == ord('c'):  # Clear text
            self.output_text = ""
            self.gesture_recognizer.clear_history()
            print("Text cleared")
        elif key == ord('s'):  # Save text
            filename = f"output_{self.frame_count}.txt"
            with open(filename, 'w') as f:
                f.write(self.output_text)
            print(f"Text saved to {filename}")
        elif key == ord('r'):  # Reset
            self.gesture_recognizer.reset_sequence()
            self.current_gesture = None
        
        return True
    
    def run(self):
        """Run the main application loop"""
        if not self.setup_camera():
            return
        
        print("Sign Language Recognition App Started")
        print("Commands:")
        print("  'q' - Quit")
        print("  'c' - Clear text")
        print("  's' - Save text to file")
        print("  'r' - Reset gesture sequence")
        print("\nPosition your hands in front of the camera to perform gestures.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Sign Language Recognition', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # If a key was pressed
                    if not self.handle_input(key):
                        break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nApplication interrupted")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.release_camera()
        cv2.destroyAllWindows()
        print("\nApplication closed")


def main():
    """Main entry point"""
    app = SignLanguageRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
