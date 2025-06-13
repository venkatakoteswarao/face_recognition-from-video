# 🎯 Face Recognition in Video using Streamlit & InsightFace

A web application that allows users to upload a **target face image** and a **video file**, and automatically detects frames where the target person appears using advanced deep learning models. The app displays similarity scores, recognition accuracy, and an output video of all matched frames.

---

## 📌 Table of Contents

- [🚀 Features](#-features)
- [🧠 Technologies & Models Used](#-technologies--models-used)
- [⚙️ How It Works](#-how-it-works)
- [📂 File Structure](#-file-structure)
- [📦 Installation](#-installation)
- [▶️ Run the App](#-run-the-app)
- [🖼️ What It Provides](#-what-it-provides)
- [📊 Example Output](#-example-output)
- [📧 Contact](#-contact)

---

## 🚀 Features

- 📷 Upload a **target image**
- 🎥 Upload a **video** file (MP4)
- 🔍 Detect the person in any frame of the video
- 📈 Display a **line graph** of similarity score over time
- ✅ Show **match accuracy** (number of matched frames vs total)
- 🎬 Play the **output video** containing matched frames

---

## 🧠 Technologies & Models Used

### 🖥️ Technologies

- **[Streamlit](https://streamlit.io/)** – UI framework for deploying ML apps
- **[InsightFace](https://github.com/deepinsight/insightface)** – Face detection and recognition library
- **OpenCV** – For reading/writing videos and frame-level annotation
- **NumPy** – For vector operations
- **Matplotlib / Streamlit line chart** – For plotting similarity graph

### 🤖 ML Models (Pre-trained from InsightFace)

| Model      | Purpose                     | Description |
|------------|-----------------------------|-------------|
| `RetinaFace` | Face Detection              | Locates faces in video frames and images |
| `ArcFace`   | Face Recognition (Embedding) | Extracts face embeddings (vector representation) to compute similarity |

Model package: `buffalo_l`

---

## ⚙️ How It Works

1. **Upload Files**  
   - User uploads a **target face image** (JPG/PNG)  
   - User uploads a **video file** (MP4)

2. **Face Embedding Extraction**  
   - Extracts face from the target image using **RetinaFace**
   - Generates 512-D embedding vector using **ArcFace**

3. **Video Frame Processing**  
   - Reads video frame by frame
   - For each frame:
     - Detects faces
     - Extracts embeddings for each face
     - Compares with target embedding using cosine similarity

4. **Matching & Filtering**  
   - If similarity score > threshold (default: **0.45**), the frame is marked as a match
   - Annotates the face with rectangle and similarity score
   - Saves matched frame to a list

5. **Results Displayed**  
   - 🎯 Shows **accuracy** (matched / total frames)
   - 📈 Plots similarity score across all frames
   - ▶️ Outputs a **video** composed of matched frames

---



