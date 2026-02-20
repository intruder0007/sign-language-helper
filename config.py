# Configuration settings for Sign Language Recognition App

# Camera settings
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Hand detection settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Gesture recognition settings
GESTURE_SEQUENCE_LENGTH = 30  # Number of frames to capture for gesture
CONFIDENCE_THRESHOLD = 0.6

# Display settings
FONT_FACE = 'Courier'
FONT_SCALE = 1.0
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # Green in BGR
HAND_COLOR = (0, 255, 0)  # Green in BGR
CONNECTION_COLOR = (255, 0, 0)  # Blue in BGR

# Supported gestures (ASL letters and numbers)
SUPPORTED_GESTURES = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
    'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
    'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
    'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
    'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y',
    'Z': 'Z', '0': '0', '1': '1', '2': '2', '3': '3',
    '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
    '9': '9', 'SPACE': ' ', 'DELETE': 'DEL', 'NOTHING': ''
}

# Model paths
MODEL_PATH = 'models/gesture_model.h5'
LANDMARKS_PATH = 'models/landmarks.pkl'

# Output text settings
OUTPUT_TEXT_MAX_LENGTH = 100
OUTPUT_TEXT_POSITION = (10, 680)
INPUT_TEXT_POSITION = (10, 650)
