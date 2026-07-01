# 🤟 Real-Time American Sign Language (ASL) Recognition

> A complete end-to-end Deep Learning project for recognizing American Sign Language alphabet signs using CNN, Transfer Learning (MobileNetV2), OpenCV, and Streamlit.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Models Built](#models-built)
- [Model Selection](#model-selection)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Learnings](#key-learnings)
- [Challenges](#challenges)
- [Future Work](#future-work)
- [Author](#author)

---

## 📖 Project Overview

This project builds a **real-time American Sign Language (ASL) alphabet recognition system** that can:
- Classify 29 ASL signs (A–Z + SPACE, DELETE, NOTHING) from images
- Predict signs in real-time using a webcam feed via OpenCV
- Provide an interactive web interface via Streamlit

The project was built entirely from scratch — no pre-written pipelines or copied code — with the goal of learning every concept behind a production-level Deep Learning image classification system.

---
## 🖼️ Screenshots

### Image Upload & Prediction
![Image Upload Prediction](screenshots/image_upload.png)

### Live Webcam Prediction
![Live Webcam](screenshots/camera_feed.png)

## 💡 Motivation

Over 70 million deaf and hard-of-hearing people worldwide use sign language as their primary mode of communication. However, the vast majority of hearing people cannot understand sign language, creating a significant communication barrier in everyday situations — hospitals, schools, workplaces, and public services.

An automated ASL recognition system can:
- Enable real-time sign language translation
- Provide accessibility features in video conferencing
- Power educational tools for learning ASL
- Support assistive technology in public and professional spaces

---

## 📂 Dataset

**Dataset:** [ASL Alphabet Dataset — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

| Property | Details |
|---|---|
| Total Classes | 29 (A–Z + space, del, nothing) |
| Training Images | 87,000 (3,000 per class) |
| Image Size | 200×200 pixels, RGB |
| Class Balance | Perfectly balanced (3,000 per class) |
| Test Images | 29 (1 per class, separate folder) |

**Why this dataset:**
- Large, balanced, and diverse enough for CNN training
- Includes practical utility classes (space, delete, nothing) for real-world word-building
- Real hand images (not synthetic) — better generalization to webcam input
- Widely benchmarked — easy to compare performance against published results

**Data Split Used:**
```
Training subset:   1,200 images/class × 29 classes = 34,800 images
Training split:    85% train / 15% validation (via ImageDataGenerator)
Test split:        100 images/class × 29 classes = 2,900 images (held out)
```

---

## 🏗️ Project Architecture

```
Raw Images (200×200 RGB)
        ↓
Image Preprocessing
(Resize → Normalize → Augment → Split)
        ↓
    ┌───────────────────────────────┐
    │   Model Training Pipeline     │
    │                               │
    │  Custom CNN  │  MobileNetV2   │
    │  (Baseline)  │  (Production)  │
    └───────────────────────────────┘
        ↓
Model Evaluation
(Accuracy, F1, Confusion Matrix)
        ↓
    ┌────────────────────────────┐
    │     Deployment Options     │
    │                            │
    │  predict_image.py          │
    │  webcam_predict.py (OpenCV)│
    │  streamlit_app.py          │
    └────────────────────────────┘
```

---

## 🧠 Models Built

### Model 1 — Custom CNN (Baseline)

Built entirely from scratch to understand every layer's role:

```
Input (64×64×3)
→ Conv2D(32, 3×3) + ReLU + MaxPooling(2×2)
→ Conv2D(64, 3×3) + ReLU + MaxPooling(2×2)
→ Conv2D(128, 3×3) + ReLU + MaxPooling(2×2)
→ Flatten
→ Dense(128) + ReLU + Dropout(0.3)
→ Dense(29) + Softmax
```

| Property | Value |
|---|---|
| Total Parameters | 686,941 (~2.62 MB) |
| Input Size | 64×64×3 |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Training Epochs | 10 (EarlyStopping) |

**Why each layer:**
- `Conv2D` — detects local spatial patterns (edges, curves, finger shapes)
- `ReLU` — introduces non-linearity, enables learning complex patterns
- `MaxPooling` — reduces spatial size, provides translation invariance
- `Flatten` — converts 3D feature maps to 1D for Dense layers
- `Dense` — combines extracted features for final classification
- `Dropout(0.3)` — randomly disables 30% of neurons to prevent overfitting
- `Softmax` — converts output to probability distribution across 29 classes

---

### Model 2 — MobileNetV2 Transfer Learning (Production Model)

Replaced the custom CNN with a pretrained MobileNetV2 backbone using a two-phase training strategy:

**Phase 1 — Feature Extraction (Frozen Base)**
```
Input (96×96×3)
→ MobileNetV2 Base (frozen, pretrained on ImageNet)
→ GlobalAveragePooling2D
→ Dense(128) + ReLU + Dropout(0.3)
→ Dense(29) + Softmax
```

**Phase 2 — Fine Tuning (Top 30 layers unfrozen)**
```
Same architecture, but top 30 MobileNetV2 layers unfrozen
Learning rate reduced to 1e-4 (10× smaller)
to prevent catastrophic forgetting of pretrained features
```

| Property | Value |
|---|---|
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Input Size | 96×96×3 |
| Phase 1 LR | 0.001 |
| Phase 2 LR | 0.0001 |
| Trainable Params (Phase 1) | ~200K (head only) |
| Trainable Params (Phase 2) | ~800K (head + top 30 layers) |

**Why MobileNetV2:**
- Pretrained on 1.2M ImageNet images — rich, generalizable visual features
- Lightweight architecture (Depthwise Separable Convolutions) — CPU-friendly inference
- GlobalAveragePooling instead of Flatten — drastically fewer parameters, better regularization
- Proven performance on hand/gesture recognition tasks

---

## 🎯 Model Selection

**Production model chosen: MobileNetV2 (Phase 2 Fine-Tuned)**

Despite both models achieving similar test accuracy, MobileNetV2 was selected for deployment based on:

| Criteria | Custom CNN | MobileNetV2 |
|---|---|---|
| Test Accuracy | 95.34% | 94.76% |
| Val Accuracy | 80.69% | 85.04% |
| Train/Val Gap | ~15% | ~10% |
| Generalization | Moderate | Better |
| Overfitting Risk | Higher | Lower |
| Industry Standard | No | Yes |

The smaller train/validation gap in MobileNetV2 (10% vs 15%) indicates **better generalization to unseen data** — more reliable in real-world deployment where input conditions vary. The custom CNN's slightly higher test accuracy is attributed to both models being evaluated on the same data distribution, where the gap disappears; in truly out-of-distribution scenarios, MobileNetV2's richer pretrained features provide more robustness.

> *"Interestingly, my custom CNN matched transfer learning performance on this specific dataset — demonstrating that for domain-specific tasks with well-balanced data, a carefully tuned architecture can be competitive with pretrained models."*

---

## 📊 Results

### Custom CNN Performance
```
Test Accuracy:        95.34%
Val Accuracy:         80.69%
Training Epochs:      10
Model Size:           2.62 MB
```

### MobileNetV2 Performance
```
Phase 1 Val Accuracy: 78.14% (frozen base)
Phase 2 Val Accuracy: 85.04% (fine-tuned)
Test Accuracy:        94.76%
Improvement over CNN: +4.35% val accuracy
Model Size:           ~9 MB
```

### Per-Class Analysis (MobileNetV2)

**Strongest classes (100% F1):**
- C, nothing, space — visually distinct, no confusion

**Weakest classes:**
| Class | Test Accuracy | Reason |
|---|---|---|
| U | 82% | Visually similar to R, X |
| N | 86% | Very similar to M (thumb position) |
| V | 87% | Similar finger configuration to R, K |
| M | 88% | Near-identical to N sign |

**Top confused pairs:**
```
U → R  (15 cases)   — similar two-finger configurations
N → M  (14 cases)   — differ only in thumb position
M → N  (12 cases)   — symmetric confusion
V → R  (8 cases)    — overlapping finger shapes
```

This confusion pattern is **consistent with documented challenges in ASL recognition research** — these letter pairs are genuinely ambiguous even to human observers in static images.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| TensorFlow 2.19 / Keras 3 | Model building and training |
| MobileNetV2 | Pretrained base model |
| OpenCV | Webcam capture and real-time inference |
| Streamlit | Web application |
| PIL / Pillow | Image preprocessing |
| NumPy | Array operations |
| Matplotlib / Seaborn | Training curves and evaluation plots |
| scikit-learn | Confusion matrix, classification report |
| Pandas | Data analysis |

---

## 📁 Project Structure

```
asl-sign-language-recognition/
├── data/                          # Dataset (not included in repo — see Dataset section)
│   ├── asl_alphabet_train/
│   ├── asl_alphabet_train_subset/ # Reduced training subset (1200/class)
│   └── asl_alphabet_test_split/   # Held-out test set (100/class)
│
├── models/
│   ├── asl_mobilenet_final.keras  # Production model (MobileNetV2)
│   ├── asl_cnn_final.keras        # Baseline model (Custom CNN)
│   └── class_indices.json         # Class name → index mapping
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb  # EDA, preprocessing, custom CNN training
│   └── 02_transfer_learning.ipynb      # MobileNetV2 transfer learning
│
├── src/
│   ├── predict_image.py           # Standalone image prediction script
│   └── webcam_predict.py          # Real-time webcam prediction (OpenCV)
│
├── app/
│   └── streamlit_app.py           # Streamlit web application
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/asl-sign-language-recognition.git
cd asl-sign-language-recognition

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow==2.19.0
opencv-python
streamlit
pillow
numpy
matplotlib
seaborn
scikit-learn
pandas
```

---

## 🚀 Usage

### 1. Predict a Single Image
```bash
python src/predict_image.py --image path/to/hand_sign.jpg
```

**Output:**
```
Loading model...
Model loaded successfully.

Image: path/to/hand_sign.jpg
Predicted Sign: A
Confidence: 97.32%
----------------------------------------
```

### 2. Predict Multiple Images
```bash
python src/predict_image.py --images img1.jpg img2.jpg img3.jpg
```

### 3. Real-Time Webcam Prediction
```bash
python src/webcam_predict.py
```

Controls:
- Place hand inside the **green rectangle**
- Hold sign steady for ~1.5 seconds to register
- **R** → reset current word
- **Q** → quit

### 4. Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```

---

## 📚 Key Learnings

**1. CNN Architecture Design**
Understanding how Conv2D filters detect edges → curves → shapes → gestures across layers, and why each architectural choice (filter size, pooling, dropout) exists.

**2. Overfitting Diagnosis**
Using train/val accuracy curves to identify overfitting early — the custom CNN showed a 15% train/val gap, confirmed by confusion matrix analysis and addressed through transfer learning.

**3. Transfer Learning Strategy**
Two-phase training: frozen base (feature extraction) followed by partial fine-tuning (top 30 layers, 10× smaller LR) to adapt ImageNet features to ASL domain without catastrophic forgetting.

**4. Domain Gap**
Discovering that a model achieving 94%+ on held-out test data from the same distribution can still struggle on out-of-distribution inputs — a fundamental real-world deployment challenge. Solutions include domain adaptation, targeted augmentation, and fine-tuning on target domain data.

**5. Evaluation Beyond Accuracy**
Using confusion matrices and per-class F1 scores to identify that most errors concentrate in linguistically similar sign pairs (M/N, U/V/R/K) — a finding consistent with ASL recognition literature, not a random model failure.

**6. Production Engineering Habits**
- Model checkpointing (never lose training progress)
- Preprocessing consistency (training vs inference must be identical)
- Saving class indices separately (critical for decoding predictions in deployment)
- Separating test set before any training (avoiding data leakage)

---

## 🚧 Challenges

**1. Hardware Constraints**
Training on CPU (Intel i5, 8GB RAM) required careful optimization:
- Reduced image size (64×64 for CNN, 96×96 for MobileNetV2)
- Reduced training subset (1,200/class instead of full 3,000)
- Batch size 32 to prevent RAM overflow
- EarlyStopping and ReduceLROnPlateau to avoid wasted compute

**2. Overfitting with Custom CNN**
Train accuracy reached 96% while validation plateaued at ~81% — a 15-point gap indicating overfitting on the reduced dataset. Addressed through transfer learning which brought the gap down to ~10%.

**3. Visually Ambiguous Signs**
Letters M/N, U/V/R/K share highly similar hand configurations and are consistently confused — not just by this model but documented across ASL recognition research. These pairs require either higher resolution input, temporal (video-based) recognition, or additional context to disambiguate reliably.

**4. Dataset vs Real World Distribution**
The training dataset uses controlled backgrounds and consistent lighting. Real-world deployment introduces variable lighting, backgrounds, and hand orientations — a classic domain gap challenge in computer vision.

---

## 🔮 Future Work

- [ ] Collect real webcam images per class and fine-tune model on target domain
- [ ] Add temporal modeling (LSTM/GRU) over video frames for dynamic signs (J, Z require motion)
- [ ] Implement MediaPipe hand landmark detection for background-invariant hand localization
- [ ] Expand to full word/sentence recognition beyond individual letters
- [ ] Optimize model with TensorFlow Lite for mobile deployment
- [ ] Add text-to-speech output for predicted words

---

## 👤 Author

**Abhishek Soni**


---

> *Built as a complete learning project — every line of code written and understood from scratch, no pre-built pipelines copied.*
