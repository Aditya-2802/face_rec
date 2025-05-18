# face_rec
## 🧠 Face Recognition System – Offline (Python + OpenCV + FaceNet)

This project uses **OpenCV** and **FaceNet (via the `face_recognition` library)** to detect and recognize faces from static images. It works completely offline and is suitable for simple identity verification or offline facial matching.



### 📌 Features

* 🔍 Detects faces from images (no live video needed)
* 🧬 Extracts facial encodings (using FaceNet/dlib)
* 🧠 Matches faces with a trained dataset
* 🖼️ Outputs labeled image with the recognized person's name
* 📂 Easy to organize your dataset by folders
* 🧑‍💻 Beginner-friendly and fully offline



### 🚀 Getting Started


#### 2️⃣ Install Requirements

Make sure you have **Python 3.8+** installed.

```bash
pip install face_recognition opencv-python
```

#### 3️⃣ Prepare Your Dataset

Create a folder structure like this:

```
dataset/
├── person1/
│   └── image1.jpg
├── person2/
│   └── image2.jpg
```

Each subfolder name (e.g. `person1`, `person2`) is used as the label during recognition.

#### 4️⃣ Encode the Faces

```bash
python encode_faces.py
```

This script scans the `dataset/` folder and saves all facial encodings into `encodings.pickle`.

#### 5️⃣ Add Test Image

Place a new image (not present in the dataset) as:

```
test.jpg
```

This will be used to recognize whose face it is.

#### 6️⃣ Recognize the Face

```bash
python recognize_face.py
```

A window will open showing:

* Detected face(s)
* The predicted name below each face

---

### 🧠 How It Works

1. **Face Detection** – Uses HOG+CNN-based model from `face_recognition`.
2. **Encoding** – Generates 128-d facial embeddings using FaceNet (via `dlib`).
3. **Comparison** – Compares embeddings using Euclidean distance.
4. **Labeling** – Draws boxes and labels matched faces.

### 🛠️ Tools & Technologies

* Python 3.10
* OpenCV
* Face Recognition (dlib + FaceNet)
* NumPy, Pickle

### 📚 What I Learned

* Preprocessing and organizing datasets for face recognition
* Encoding and comparing faces using embeddings
* Visualizing output with OpenCV
* Building offline AI tools using Python




