# 🚂 Railway Fault Detector

An AI-powered web application for detecting structural defects in railway tracks using deep learning (EfficientNetB3 CNN). Upload a track image and get an instant defective/non-defective classification with confidence scores.

---

## 📸 Demo

> Upload a railway track photo → the model predicts **Defective** or **Non defective** with a confidence score and inference time.

---

## 🧠 How It Works

1. A **CNN model (EfficientNetB3)** is trained on labeled railway track images.
2. The model outputs a **sigmoid probability** — P(Non defective).
3. A **tuned decision threshold** (default: 0.50) converts the probability into a binary class.
4. Optional **probability calibration** rescales the raw score relative to the decision boundary for more meaningful confidence display.

---

## 📁 Project Structure

```
Railway/
├── app.py                              # Streamlit web application
├── best_model.keras                    # Trained EfficientNetB3 model
├── improved_model_accuracy_clean.ipynb # Training & evaluation notebook
├── requirements.txt                    # Python dependencies
└── dataset/
    ├── Train/
    │   ├── Defective/
    │   └── Non defective/
    ├── Validation/
    │   ├── Defective/
    │   └── Non defective/
    └── Test/
        ├── Defective/
        └── Non defective/
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Railway
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Model loading & inference |
| `streamlit` | Web application framework |
| `scikit-learn` | Evaluation metrics |
| `matplotlib` | Plotting training curves |
| `jupyter` | Training notebook |

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| **Model Selection** | Switch between available trained models |
| **Decision Threshold Slider** | Fine-tune the Defective vs Non defective cutoff (tuned on validation set) |
| **Aspect Ratio Crop** | Center-crop images to avoid squashing during resize |
| **Probability Calibration** | Scales raw probabilities relative to the decision boundary |
| **Raw Probability View** | Toggle to see exact class-level sigmoid output |
| **Uncertainty Warning** | Flags predictions close to the decision boundary (< 8% margin) |

---

## 🗂️ Dataset

Images are organized into **Train / Validation / Test** splits with two classes:

- 🔴 **Defective** — tracks with visible structural faults
- 🟢 **Non defective** — tracks in good condition

Images were captured in the field (JPEG, from a mobile camera). EXIF orientation is automatically corrected during preprocessing.

---

## 🔬 Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNetB3 |
| Input size | 224 × 224 px |
| Output | Sigmoid (binary) |
| Loss | Binary cross-entropy |
| Classes | Defective, Non defective |
| Decision threshold | 0.50 (tuned on validation set) |

> **Note:** The model's rescaling/normalization layers are baked in — raw pixel values [0, 255] are passed directly; do **not** divide by 255 before inference.

---

## 🚀 Training

Open and run the notebook to retrain or evaluate the model:

```bash
jupyter notebook improved_model_accuracy_clean.ipynb
```

The notebook covers:
- Data loading with `tf.keras.utils.image_dataset_from_directory`
- Transfer learning from EfficientNetB3 pretrained weights
- Fine-tuning, augmentation, and learning rate scheduling
- Validation threshold tuning
- Confusion matrix and classification report

---

## 📄 License

This project is for academic and research purposes.
