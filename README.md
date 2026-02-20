# Sign Language Recognition App

A real-time Python application that uses OpenCV and MediaPipe to recognize American Sign Language (ASL) gestures and convert them into text.

## ✨ Features

- **Real-time Hand Detection**: Detects and tracks hands using MediaPipe
- **Gesture Recognition**: Recognizes ASL letters (A-Z), numbers (0-9), and special commands
- **Live Video Processing**: Processes video frames in real-time with performance optimization
- **Customizable Model**: Train your own gesture recognition model using collected data
- **Easy To Use**: Simple keyboard controls and intuitive interface
- **Data Collection**: Built-in tool to collect training data for custom gestures

## 🎯 Enhanced Features (v2.0)

### Advanced Visualization
- Real-time confidence bars for hand detection
- Gesture recognition confidence display
- Gesture history trail tracking
- Performance metrics display (FPS, detection time, accuracy)
- Gesture recognition grid (shows recognized vs available)
- Enhanced hand skeleton with finger identification

### Statistics & Analytics
- Session statistics tracking
- Per-gesture performance metrics
- FPS monitoring and framerate analysis
- Performance timing (detection & recognition)
- Automatic improvement recommendations
- JSON export for data analysis

### Voice Feedback
- Text-to-speech for recognized characters
- Configurable speech rate and volume
- Sound effects for gesture confirmation
- Toggle voice on/off during operation

### Intelligent Analysis
- Hand movement stability analysis
- Gesture speed and motion detection
- Hand size tracking and analysis
- Gesture similarity comparison
- Gesture pattern recognition

### Data Preprocessing
- Landmark normalization (Z-score, min-max, robust)
- Movement smoothing (Savitzky-Golay filter)
- Outlier detection and removal
- Hand shape descriptor extraction
- Data augmentation for training

## Project Structure

```
sign-language-helper/
├── src/
│   ├── config.py                 # Configuration settings
│   ├── hand_detector.py          # Hand detection using MediaPipe
│   ├── gesture_recognizer.py     # Gesture recognition model
│   ├── main.py                   # Main application
│   ├── data_collector.py         # Training data collection tool
│   └── train_model.py            # Model training script
├── models/                        # Trained model files
├── data/                          # Training data directory
├── tests/                         # Test files
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

## Requirements

- Python 3.8+
- Webcam/Camera
- Windows, macOS, or Linux

## Installation

1. **Clone/Download the project**:
   ```bash
   cd "sign language helper"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start (Demo Mode)

To run the app in demo mode without a trained model:

```bash
cd src
python main.py
```

### Enhanced App (With Advanced Features)

For the enhanced version with visualization, statistics, and voice feedback:

```bash
cd src
python enhanced_main.py
```

**Enhanced App Features:**
- Real-time performance metrics
- Voice feedback for recognized gestures
- Session statistics tracking
- Automatic improvement recommendations
- Advanced visualization

### Training a Custom Model

1. **Collect Training Data**:
   ```bash
   cd src
   python data_collector.py
   ```
   - Choose the gesture you want to collect data for
   - Position your hand and press SPACE to capture samples
   - Recommend collecting 100+ samples per gesture

2. **Train the Model**:
   ```bash
   python train_model.py
   ```
   - The script will load all collected data
   - Train a neural network on the data
   - Save the trained model to `models/gesture_model.h5`

3. **Use the Trained Model**:
   ```bash
   python main.py
   # or for enhanced version:
   python enhanced_main.py
   ```
   - The app will automatically load your trained model
   - Gestures will be recognized and converted to text

## Keyboard Shortcuts

### Basic App (main.py)

| Key | Function |
|-----|----------|
| `q` | Quit the application |
| `c` | Clear recognized text |
| `s` | Save text to file |
| `r` | Reset gesture sequence |

### Enhanced App (enhanced_main.py)

| Key | Function |
|-----|----------|
| `q` | Quit the application |
| `c` | Clear recognized text |
| `s` | Save text to file |
| `r` | Reset gesture sequence |
| `v` | Toggle voice feedback on/off |
| `t` | Toggle statistics display |
| `x` | Export session statistics to JSON |

## Supported Gestures

### Letters
- A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

### Numbers
- 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

### Special Commands
- `SPACE`: Add space character
- `DELETE`: Remove last character
- `NOTHING`: No gesture (idle)

## Configuration

Edit `src/config.py` to customize:

- **Camera Settings**: Resolution, FPS, camera ID
- **Detection Confidence**: Adjust hand detection sensitivity
- **Recognition Threshold**: Confidence threshold for gesture acceptance
- **Gesture Sequence Length**: Number of frames to use for recognition

## How It Works

1. **Hand Detection**: Uses MediaPipe Hands to detect 21 landmarks per hand
2. **Feature Extraction**: Extracts normalized features from hand landmarks
3. **Temporal Sequencing**: Collects features over 30 frames (1 second)
4. **Neural Network**: 3-layer neural network classifies the gesture
5. **Text Conversion**: Maps recognized gesture to corresponding character

## Performance Tips

- Ensure good lighting for better hand detection
- Keep hands clearly visible in the frame
- Move slowly and deliberately when performing gestures
- Maintain consistent gesture recognition by training with varied angles and distances
- Use both static and dynamic gestures for better accuracy

## Troubleshooting

### Camera not opening?
- Check if another application is using the camera
- Update the `CAMERA_ID` in `config.py` (try 0, 1, 2, etc.)
- Ensure camera permissions are granted

### Low recognition accuracy?
- Collect more training data (minimum 100+ samples per gesture)
- Make sure lighting is adequate
- Perform gestures consistently (similar speed, position, size)
- Increase `GESTURE_SEQUENCE_LENGTH` in `config.py`

### Model not loading?
- Ensure model file exists in `models/` directory
- Train a new model using `train_model.py`
- Check file permissions

## Model Architecture

The gesture recognition model uses a 3-layer neural network:

```
Input (84 features)
    ↓
Dense(128, relu) + Dropout(0.3)
    ↓
Dense(64, relu) + Dropout(0.3)
    ↓
Dense(32, relu)
    ↓
Dense(29, softmax)  # 29 gesture classes
    ↓
Output (probability distribution)
```

## Future Enhancements

- [ ] Support for more gesture vocabularies (e.g., BSL, LSF)
- [ ] Continuous sign tracking without temporal windowing
- [ ] Real-time emotion/expression recognition
- [ ] Integration with text-to-speech for output
- [ ] Mobile application version
- [ ] Pre-trained models for common gestures
- [ ] Multi-hand gesture support for complex signs

## License

This project is open source and available for educational and personal use.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [American Sign Language (ASL)](https://www.lifeprint.com/)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the code comments and docstrings
- Experiment with configuration settings in `config.py`

---

**Created**: 2026  
**Version**: 1.0  
**Status**: Active Development
