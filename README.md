# RoadGuard AI 🚗

## Driver Drowsiness Detection Using Computer Vision and Deep Learning

RoadGuard AI is a real-time driver drowsiness detection system
that analyzes the driver's eye and mouth states using computer vision
and deep learning.

## Project Overview

The system uses two lightweight deep learning models:

- Eye Model: Detects Open Eyes / Closed Eyes.
- Mouth Model: Detects Yawn / No Yawn.

The predictions from both models are combined to determine the
driver's drowsiness state.

## System Architecture

The system follows these main steps:

1. Capture video frames from a camera.
2. Detect the driver's face using OpenCV YuNet.
3. Extract facial landmarks.
4. Extract Eye and Mouth Regions of Interest (ROIs).
5. Pass the Eye ROI to the Eye Model.
6. Pass the Mouth ROI to the Mouth Model.
7. Combine both predictions using decision logic.
8. Generate a drowsiness alert when necessary.

## AI Models

### Eye Model

The Eye Model performs binary classification:

- Open Eyes
- Closed Eyes

### Mouth Model

The Mouth Model performs binary classification:

- Yawn
- No Yawn

Both models use Transfer Learning with MobileNetV2 and Fine-Tuning.

## Dataset

The project uses the Kaggle Drowsiness Dataset.

The data is divided into:

- Training set
- Validation set
- Test set

Data augmentation is applied to reduce overfitting.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- YuNet
- NumPy
- Matplotlib
- Scikit-learn

## Model Training

The models are trained using:

- Adam Optimizer
- Binary Crossentropy Loss
- Accuracy Metric
- Early Stopping

Fine-tuning is performed using a small learning rate of 1e-5.

## Evaluation

The models are evaluated using:

- Test Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

## Project Structure

```text
RoadGuard-AI/
├── models/
├── src/
├── notebooks/
├── data/
├── results/
├── presentation/
├── README.md
├── requirements.txt
└── .gitignore
