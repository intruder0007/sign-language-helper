"""
Sign Language Recognition App
A Python-based application for real-time sign language gesture recognition
"""

__version__ = "1.0.0"
__author__ = "Sign Language Helper Team"
__description__ = "Real-time sign language gesture recognition using OpenCV and MediaPipe"

from .hand_detector import HandDetector
from .gesture_recognizer import GestureRecognizer

__all__ = ['HandDetector', 'GestureRecognizer']
