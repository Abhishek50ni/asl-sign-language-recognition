"""
webcam_predict.py
Real-time ASL sign recognition using webcam and MobileNetV2.

Usage:
    python src/webcam_predict.py

Controls:
    Q → quit
    R → reset current word
"""

import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
# Set working directory to project root so relative paths work correctly
os.chdir(r"C:\Users\1abhi\Downloads\NON-AARTI\Sign_Language_to_text&Audio")
# ---------------------------------------------------------
# Configuration — must match MobileNetV2 training exactly
# ---------------------------------------------------------
IMG_SIZE         = 96
MODEL_PATH       = "models/asl_mobilenet_final.keras"
CLASS_INDICES_PATH = "models/class_indices.json"
CONFIDENCE_THRESHOLD = 0.75 # only accept predictions above 75% confidence
                              # prevents noisy/uncertain predictions cluttering output


def load_class_mapping(path):
    """Invert class_indices: {name: idx} -> {idx: name}"""
    with open(path, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}


def preprocess_roi(roi):
    """
    Preprocesses the webcam ROI for model prediction.
    roi: numpy array (BGR, from OpenCV)
    Returns: normalized numpy array ready for model input
    """
    # Convert BGR (OpenCV default) to RGB (model trained on RGB)
    roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for Keras-compatible resizing
    pil_img = Image.fromarray(roi_rgb)
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array and normalize
    img_array = img_to_array(pil_img) / 255.0

    # Add batch dimension: (96,96,3) -> (1,96,96,3)
    return np.expand_dims(img_array, axis=0)


def predict_sign(model, index_to_class, roi):
    """
    Runs model prediction on preprocessed ROI.
    Returns predicted class name and confidence.
    """
    preprocessed = preprocess_roi(roi)
    predictions = model.predict(preprocessed, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    predicted_class = index_to_class[predicted_index]
    return predicted_class, confidence


def draw_ui(frame, predicted_class, confidence, current_word, roi_x, roi_y, roi_size):
    """
    Draws the UI overlay on the webcam frame:
    - ROI rectangle (where to place hand)
    - Predicted letter
    - Confidence bar
    - Current word being built
    """
    # Draw ROI rectangle (green guide box)
    cv2.rectangle(frame,
                  (roi_x, roi_y),
                  (roi_x + roi_size, roi_y + roi_size),
                  (0, 255, 0), 2)

    # Predicted letter (large, top left)
    cv2.putText(frame,
                f"Sign: {predicted_class}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3)

    # Confidence percentage
    conf_pct = int(confidence * 100)
    cv2.putText(frame,
                f"Confidence: {conf_pct}%",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 0), 2)

    # Confidence bar (visual)
    bar_width = int(confidence * 200)  # max 200px wide
    cv2.rectangle(frame, (10, 115), (10 + bar_width, 135), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, 115), (210, 135), (255, 255, 255), 1)

    # Current word being built
    cv2.putText(frame,
                f"Word: {current_word}",
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    # Controls hint
    cv2.putText(frame,
                "Q: Quit | R: Reset word",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    return frame


def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    index_to_class = load_class_mapping(CLASS_INDICES_PATH)
    print("Model loaded. Starting webcam...")

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # 0 = default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ROI settings — center-left region where user places hand
    roi_x, roi_y, roi_size = 50, 50, 400  # x, y, size of square ROI

    current_word = ""          # builds up word letter by letter
    last_predicted = ""        # tracks last prediction to avoid duplicates
    stable_count = 0           # counts consecutive identical predictions
    STABLE_THRESHOLD = 15      # require 15 stable frames before accepting letter
                               # at ~10fps this is ~1.5 seconds of holding the sign

    print("Webcam ready. Place hand in the green box.")
    print("Hold a sign steady for ~1.5 seconds to register a letter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # Flip frame horizontally (mirror effect — more natural for user)
        frame = cv2.flip(frame, 1)

        # Extract ROI from frame
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

        # Predict on ROI
        predicted_class, confidence = predict_sign(model, index_to_class, roi)

        # Stability logic: only register a letter after STABLE_THRESHOLD
        # consecutive frames predict the same class above confidence threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            if predicted_class == last_predicted:
                stable_count += 1
            else:
                stable_count = 0
                last_predicted = predicted_class

            # Register letter after stable prediction
            if stable_count == STABLE_THRESHOLD:
                if predicted_class == "space":
                    current_word += " "
                elif predicted_class == "del":
                    current_word = current_word[:-1]  # delete last character
                elif predicted_class != "nothing":
                    current_word += predicted_class
                stable_count = 0  # reset after registering

        # Draw UI overlay
        frame = draw_ui(frame, predicted_class, confidence,
                       current_word, roi_x, roi_y, roi_size)

        # Show frame
        cv2.imshow("ASL Sign Recognition", frame)

        # Keypress handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):     # Q → quit
            break
        elif key == ord('r'):   # R → reset word
            current_word = ""
            print("Word reset.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()