"""
Utility functions for the Sign Language Recognition App
"""

import os
import json
from pathlib import Path
from datetime import datetime


def create_output_directory(base_path='./output'):
    """
    Create output directory for saving results
    
    Args:
        base_path: Base path for output directory
        
    Returns:
        output_dir: Path to created directory
    """
    output_dir = Path(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_text_to_file(text, filename=None, directory='./output'):
    """
    Save recognized text to a file
    
    Args:
        text: Text to save
        filename: Output filename (auto-generated if None)
        directory: Directory to save file in
        
    Returns:
        filepath: Full path to saved file
    """
    output_dir = create_output_directory(directory)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recognized_text_{timestamp}.txt"
    
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(text)
    
    print(f"Text saved to: {filepath}")
    return filepath


def load_text_from_file(filepath):
    """
    Load text from a file
    
    Args:
        filepath: Path to the file
        
    Returns:
        text: Content of the file
    """
    try:
        with open(filepath, 'r') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None


def export_session_data(text, gestures, directory='./output', session_name=None):
    """
    Export session data including recognized text and gesture history
    
    Args:
        text: Recognized text
        gestures: List of recognized gestures with confidence scores
        directory: Output directory
        session_name: Custom session name
        
    Returns:
        filepath: Path to saved JSON file
    """
    output_dir = create_output_directory(directory)
    
    if session_name is None:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'recognized_text': text,
        'gesture_history': gestures,
        'total_gestures': len(gestures)
    }
    
    filepath = output_dir / f"{session_name}.json"
    
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    print(f"Session data saved to: {filepath}")
    return filepath


def calculate_recognition_statistics(gestures):
    """
    Calculate statistics about gesture recognition
    
    Args:
        gestures: List of gesture dictionaries with 'gesture' and 'confidence' keys
        
    Returns:
        stats: Dictionary with statistical information
    """
    if not gestures:
        return {
            'total_gestures': 0,
            'average_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0
        }
    
    confidences = [g['confidence'] for g in gestures]
    
    stats = {
        'total_gestures': len(gestures),
        'average_confidence': sum(confidences) / len(confidences),
        'min_confidence': min(confidences),
        'max_confidence': max(confidences),
        'unique_gestures': len(set(g['gesture'] for g in gestures))
    }
    
    return stats


def print_statistics(stats):
    """
    Print gesture recognition statistics
    
    Args:
        stats: Statistics dictionary
    """
    print("\n" + "="*50)
    print("Gesture Recognition Statistics")
    print("="*50)
    print(f"Total Gestures Recognized: {stats['total_gestures']}")
    print(f"Unique Gestures: {stats['unique_gestures']}")
    print(f"Average Confidence: {stats['average_confidence']:.2%}")
    print(f"Min Confidence: {stats['min_confidence']:.2%}")
    print(f"Max Confidence: {stats['max_confidence']:.2%}")
    print("="*50 + "\n")


def validate_camera_available():
    """
    Check if camera is available
    
    Returns:
        available: True if camera is available
    """
    import cv2
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return available


def get_available_cameras():
    """
    Get list of available camera indices
    
    Returns:
        cameras: List of available camera indices
    """
    import cv2
    cameras = []
    
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    
    return cameras


def benchmark_hand_detection(num_frames=100):
    """
    Benchmark hand detection performance
    
    Args:
        num_frames: Number of frames to process
        
    Returns:
        fps: Average frames per second
    """
    import cv2
    import time
    from hand_detector import HandDetector
    
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not available")
        return 0
    
    start_time = time.time()
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, _ = detector.detect_hands(frame)
        frame_count += 1
    
    cap.release()
    
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    print(f"\nBenchmark Results:")
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    return fps
