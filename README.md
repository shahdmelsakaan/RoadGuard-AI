# RoadGuard AI 🚗

## Driver Drowsiness Detection Using Computer Vision and Deep Learning

RoadGuard AI is an AI-based driver drowsiness detection system that analyzes the driver's eye and mouth states to detect possible signs of drowsiness.

The system uses two deep learning models:

- Eye Model: Detects Closed Eyes / Open Eyes.
- Mouth Model: Detects No Yawn / Yawn.

The predictions from both models are combined to determine the driver's final state.

---

## Problem

Driver drowsiness is a major safety problem because fatigue can reduce a driver's reaction time and increase the risk of accidents.

RoadGuard AI aims to provide an automated system that can analyze facial features and identify possible signs of driver drowsiness.

---

## Project Workflow

The system follows these main steps:

1. Load and preprocess the eye and mouth datasets.
2. Train the Eye and Mouth deep learning models.
3. Apply Transfer Learning using MobileNetV2.
4. Fine-tune the models on the drowsiness dataset.
5. Detect the driver's face using YuNet.
6. Extract the Eye and Mouth Regions of Interest (ROIs).
7. Classify the eye and mouth states.
8. Combine the predictions using decision logic.
9. Generate the final drowsiness state.

---

## AI Models

### Eye Model

The Eye Model performs binary classification:

- Closed Eyes
- Open Eyes

The eye images are resized to 64 × 64 and processed as grayscale images.

### Mouth Model

The Mouth Model performs binary classification:

- No Yawn
- Yawn

The mouth images are resized to 64 × 64 and processed as RGB images.

---

## Deep Learning Architecture

The project uses MobileNetV2 as a pre-trained backbone with ImageNet weights.

A custom classification head is added using:

- Global Average Pooling
- Dense layer with 128 neurons
- ReLU activation
- Dropout with 50%
- Sigmoid output layer

Transfer Learning and Fine-Tuning are used to adapt the pre-trained models to the drowsiness detection task.

---

## Dataset

The project uses the Drowsiness Dataset from Kaggle.

The dataset is divided into:

- Training set
- Validation set
- Test set

The dataset contains images for both eye-state and mouth-state classification.

---

## Data Preprocessing

Images are resized to:

64 × 64 pixels

The project uses MobileNetV2 preprocessing.

Data augmentation is also applied using:

- Random Horizontal Flip
- Random Rotation
- Random Zoom

These techniques help improve model generalization and reduce overfitting.

---

## Model Training

The models are trained using:

- Adam Optimizer
- Binary Crossentropy Loss
- Accuracy Metric
- Early Stopping

Fine-tuning is performed using a small learning rate of:

1e-5

The last 30 layers of the pre-trained backbone are fine-tuned while earlier layers remain frozen.

---

## Model Evaluation

The models are evaluated using the test dataset.

The evaluation includes:

- Test Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

## Face Detection

The system uses OpenCV YuNet for face detection.

YuNet detects the driver's face and provides facial landmarks.

These landmarks are used to extract:

- Eye Region of Interest
- Mouth Region of Interest

The extracted regions are then passed to the corresponding AI models.

---

## Drowsiness Decision Logic

The final state is determined using the Eye and Mouth predictions:

| Eye State | Mouth State | Final Result |
|-----------|-------------|--------------|
| Closed | Yawn | DROWSY |
| Closed | No Yawn | POSSIBLE DROWSINESS |
| Open | Yawn | POSSIBLE DROWSINESS |
| Open | No Yawn | ALERT |

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- YuNet
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Structure

```text
RoadGuard-AI/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
├── models/
└── presentation/


## Pre-trained Models

The trained models are not included directly in this repository because of file size limitations.

The project uses:

- Eye Model: `eye_model.keras` https://drive.google.com/file/d/16flKjMzD1wzE-gAWv9FKXiCCoJQkQ54s/view?usp=drive_link
- Mouth Model: `mouth_model.keras`   https://drive.google.com/file/d/1GNjbvq1Wty7vv_XUQSz0w1TguwIcnTYL/view?usp=drive_link
- Face Detector: `face_detection_yunet_2023mar.onnx`    https://drive.google.com/file/d/1FUaHh3XB8J5RsAb2IYsM1hPrqLBzZqOJ/view?usp=drive_link
