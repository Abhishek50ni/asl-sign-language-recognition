"""
predict_image.py
Standalone script to predict ASL alphabet sign from one or more images.

Usage:
    python predict_image.py --image path/to/image.jpg
    python predict_image.py --images path/to/img1.jpg path/to/img2.jpg
"""

import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------------------------------------------------
# Configuration — must match training preprocessing exactly
# ---------------------------------------------------------
MODEL_PATH = "models/asl_mobilenet_final.keras"
IMG_SIZE = 96  # update this too — must match MobileNetV2 training size
CLASS_INDICES_PATH = "models/class_indices.json"


def load_class_mapping(path):
    """
    Loads class_indices.json (class_name -> index) and inverts it
    to index -> class_name, since the model predicts indices but
    we need human-readable letters as output.
    """
    with open(path, "r") as f:
        class_indices = json.load(f)
    # Invert dictionary: {'A': 0, 'B': 1, ...} -> {0: 'A', 1: 'B', ...}
    index_to_class = {v: k for k, v in class_indices.items()}
    return index_to_class


def preprocess_image(image_path):
    """
    Loads and preprocesses a single image to match training pipeline:
    - Resize to IMG_SIZE x IMG_SIZE
    - Convert to array
    - Normalize pixel values to [0, 1]
    - Add batch dimension (model expects shape: (batch, H, W, C))
    """
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))  # resize identical to training
    img_array = img_to_array(img)            # convert PIL image -> numpy array, shape (64,64,3)
    img_array = img_array / 255.0            # normalize identical to training (rescale=1./255)
    img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1,64,64,3) - batch of 1
    return img_array


def predict_single_image(model, index_to_class, image_path):
    """
    Predicts the ASL class for a single image and prints result with confidence.
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)  # shape (1, 29) - probability per class

    predicted_index = np.argmax(predictions[0])         # index of highest probability
    predicted_class = index_to_class[predicted_index]    # convert index -> letter
    confidence = predictions[0][predicted_index] * 100   # convert to percentage

    print(f"Image: {image_path}")
    print(f"Predicted Sign: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 40)

    return predicted_class, confidence


def predict_multiple_images(model, index_to_class, image_paths):
    """
    Runs prediction on a list of images, one by one.
    """
    results = []
    for path in image_paths:
        predicted_class, confidence = predict_single_image(model, index_to_class, path)
        results.append((path, predicted_class, confidence))
    return results


def main():
    parser = argparse.ArgumentParser(description="ASL Sign Prediction Script")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--images", type=str, nargs="+", help="Paths to multiple images")
    args = parser.parse_args()

    if not args.image and not args.images:
        print("Error: provide either --image or --images")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)
    index_to_class = load_class_mapping(CLASS_INDICES_PATH)
    print("Model loaded successfully.\n")

    if args.image:
        predict_single_image(model, index_to_class, args.image)

    if args.images:
        predict_multiple_images(model, index_to_class, args.images)


if __name__ == "__main__":
    main()