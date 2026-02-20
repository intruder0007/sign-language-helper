"""
Model Training Module
Train gesture recognition model on collected data
"""

import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data_collector import GestureDataCollector
from gesture_recognizer import GestureRecognizer
import config


class ModelTrainer:
    """
    Trains gesture recognition model on collected data
    """
    
    def __init__(self):
        """Initialize the trainer"""
        self.data_collector = GestureDataCollector()
        self.gesture_recognizer = GestureRecognizer()
        self.label_encoder = LabelEncoder()
    
    def train_model(self, test_size=0.2):
        """
        Train model on collected gesture data
        
        Args:
            test_size: Fraction of data to use for testing
        """
        print("Loading collected gesture data...")
        X, y = self.data_collector.load_all_data()
        
        if X is None or len(X) == 0:
            print("Error: No training data found. Please collect data first using data_collector.py")
            return False
        
        print(f"Loaded {len(X)} samples")
        print(f"Feature size per sample: {X[0].shape}")
        print(f"Unique gestures: {len(set(y))}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Handle feature size mismatch
        # Standardize feature size to 84 (21 landmarks * 3 coords + 20 distances)
        expected_size = 84
        if X.shape[1] != expected_size:
            print(f"Warning: Feature size mismatch. Expected {expected_size}, got {X.shape[1]}")
            print("Padding or truncating features...")
            
            if X.shape[1] < expected_size:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_size - X.shape[1]))
                X = np.concatenate([X, padding], axis=1)
            else:
                # Truncate
                X = X[:, :expected_size]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create and train model
        print("\nTraining gesture recognition model using RandomForest...")
        self.gesture_recognizer.train_on_data(
            X_train, y_train, epochs=100, batch_size=16
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        y_pred = self.gesture_recognizer.model.predict(X_test)
        test_accuracy = np.mean(y_pred == y_test)
        
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Save model
        model_path = Path(config.MODEL_PATH)
        model_path.parent.mkdir(exist_ok=True)
        self.gesture_recognizer.save_model(str(model_path))
        
        # Save label encoder
        encoder_path = Path(config.LANDMARKS_PATH)
        encoder_path.parent.mkdir(exist_ok=True)
        with open(str(encoder_path), 'wb') as f:
            import pickle
            pickle.dump(self.label_encoder, f)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Label encoder saved to: {encoder_path}")
        
        return True


def main():
    """Main entry point for training"""
    print("\n=== Sign Language Gesture Recognition Model Trainer ===\n")
    
    trainer = ModelTrainer()
    success = trainer.train_model()
    
    if success:
        print("\n✓ Model training completed successfully!")
        print("You can now use the trained model in the main application.")
    else:
        print("\n✗ Model training failed.")


if __name__ == "__main__":
    main()

