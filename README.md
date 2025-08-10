# Shape Detector – Hebb Neural Network

A **Streamlit** web application for detecting simple geometric shapes (**rectangles** and **triangles**) using a basic **Hebbian learning neural network**.

---

## ⚡ Quick Start

```bash
# 1️⃣ Clone the repo
git clone https://github.com/your-username/shape-detector.git
cd shape-detector

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the app
streamlit run app.py
```

Open the link Streamlit provides (usually http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
project-folder/
│
├── shapes/
│   ├── rectangles/    # Training & testing images of rectangles (labeled as 1)
│   ├── triangles/     # Training & testing images of triangles (labeled as -1)
│   ├── mixed/         # Unlabeled images for prediction (filenames contain 1 or -1 at the end)
│
├── common.py          # Helper functions for network logic
├── app.py             # Main Streamlit application
├── README.md          # You are here
└── requirements.txt   # Python dependencies
```

---

## 📦 Requirements

### 1. Python version

Make sure you have **Python 3.9+** installed.

You can check your version:

```bash
python --version
```

---

### 2. Install dependencies

First, create a **virtual environment** (recommended):

```bash
python -m venv venv
```

Activate it:

**Windows (PowerShell)**:

```powershell
venv\Scripts\activate
```

**Mac/Linux**:

```bash
source venv/bin/activate
```

Then install all required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App Locally

```bash
cd project-folder
streamlit run app.py
```

---

## 🖥 How to Use the App

1. **Train Neural Net** – Train on labeled triangles and rectangles.
2. **Test Neural Net with Trained Data** – Test accuracy and see results table.
3. **Predict Shapes** – Classify shapes from `shapes/mixed`.
4. **Clear Data** – Clear displayed results without retraining.

---

## 📌 Notes

- Images should be **50×50 grayscale** (white background, black shape).
- Mixed image filenames should end with `_1` for rectangles, `_-1` for triangles.  
  Example:
  ```
  rectangle_12_1.bmp
  triangle_03_-1.bmp
  ```

---

## 📜 License

MIT License — free to use, modify, and share.
