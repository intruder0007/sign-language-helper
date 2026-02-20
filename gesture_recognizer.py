"""
Gesture Recognition Module
Recognizes sign language gestures from hand landmarks
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from config import GESTURE_SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, SUPPORTED_GESTURES


class GestureRecognizer:
    """
    Recognizes sign language gestures from hand landmarks
    """
    
    def __init__(self, model_path=None):
        """
        Initialize gesture recognizer
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.gesture_sequence = []
        self.gesture_history = []
        self.model = None
        self.gesture_classes = list(SUPPORTED_GESTURES.keys())
        self.label_encoder = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._init_default_model()
    
    def _init_default_model(self):
        """Initialize a simple default model using RandomForest"""
        # Create a RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        # Initialize with dummy data to prepare the model
        self.is_trained = False
    
    def load_model(self, model_path):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from: {model_path}")
            self.is_trained = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self._init_default_model()
    
    def save_model(self, model_path):
        """
        Save the trained model
        
        Args:
            model_path: Path to save the model
        """
        if self.model:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to: {model_path}")
    
    def add_frame_features(self, features):
        """
        Add frame features to gesture sequence
        
        Args:
            features: Feature vector from current frame
        """
        if features is not None:
            self.gesture_sequence.append(features)
            
            # Keep only recent frames
            if len(self.gesture_sequence) > GESTURE_SEQUENCE_LENGTH:
                self.gesture_sequence.pop(0)
    
    def reset_sequence(self):
        """Reset gesture sequence"""
        self.gesture_sequence = []
    
    def recognize_gesture(self):
        """
        Recognize gesture from collected hand landmarks
        
        Returns:
            gesture: Recognized gesture label
            confidence: Confidence score
        """
        if len(self.gesture_sequence) < GESTURE_SEQUENCE_LENGTH // 2:
            return None, 0.0
        
        try:
            # Average the features across frames
            features_array = np.array(self.gesture_sequence)
            averaged_features = np.mean(features_array, axis=0).reshape(1, -1)
            
            if self.model is None or not hasattr(self.model, 'predict_proba'):
                return 'UNKNOWN', 0.0
            
            # Predict gesture
            predictions = self.model.predict_proba(averaged_features)[0]
            confidence = np.max(predictions)
            gesture_idx = np.argmax(predictions)
            
            if confidence >= CONFIDENCE_THRESHOLD:
                gesture = self.model.classes_[gesture_idx]
                return gesture, confidence
            else:
                return None, confidence
        
        except Exception as e:
            print(f"Error during recognition: {e}")
            return None, 0.0
    
    def train_on_data(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the model on custom data
        
        Args:
            X_train: Training features
            y_train: Training labels (encoded as integers)
            epochs: Number of training epochs (ignored for RandomForest)
            batch_size: Batch size (ignored for RandomForest)
        """
        if self.model is None:
            self._init_default_model()
        
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            print("Model training completed")
        except Exception as e:
            print(f"Error during training: {e}")
    
    def add_to_history(self, gesture, confidence):
        """
        Add recognized gesture to history
        
        Args:
            gesture: Gesture label
            confidence: Confidence score
        """
        if gesture:
            self.gesture_history.append({
                'gesture': gesture,
                'confidence': confidence
            })
    
    def get_gesture_text(self):
        """
        Get text representation of recognized gestures
        
        Returns:
            text: String of recognized characters
        """
        text = ""
        for item in self.gesture_history:
            gesture = item['gesture']
            if gesture in SUPPORTED_GESTURES:
                text += SUPPORTED_GESTURES[gesture]
            elif gesture == 'DELETE':
                text = text[:-1]  # Remove last character
        return text
    
    def clear_history(self):
        """Clear gesture history"""
        self.gesture_history = []

