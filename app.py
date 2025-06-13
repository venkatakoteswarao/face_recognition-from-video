import streamlit as st
import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from tempfile import NamedTemporaryFile

# Title
st.title("🎯 Face Recognition in Video using InsightFace")

# Upload target image
target_file = st.file_uploader("📷 Upload Target Face Image (jpg/png)", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("🎥 Upload Video File (mp4)", type=["mp4"])

# Proceed only if both files are uploaded
if target_file and video_file:
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        temp_img.write(target_file.read())
        target_image_path = temp_img.name

    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
        temp_vid.write(video_file.read())
        video_path = temp_vid.name

    output_folder = "matched_frames"
    output_video = "matched_faces_output.mp4"
    os.makedirs(output_folder, exist_ok=True)

    # InsightFace setup
    face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0)

    # Load target image and get embedding
    target_image = cv2.imread(target_image_path)
    target_faces = face_app.get(target_image)
    if not target_faces:
        st.error("❌ No face detected in the target image.")
        st.stop()
    target_feature = target_faces[0].embedding

    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    matched_frames = []
    similarity_scores = []
    total_frames = 0
    matched_count = 0

    st.info("🔍 Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        total_frames += 1

        faces = face_app.get(frame)
        matched = False
        for face in faces:
            bbox = face.bbox.astype(int)
            similarity = np.dot(face.embedding, target_feature) / (
                np.linalg.norm(face.embedding) * np.linalg.norm(target_feature)
            )
            similarity_scores.append((frame_num, similarity))

            if similarity > 0.45 and not matched:
                matched = True
                matched_count += 1
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{similarity:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                matched_frames.append(frame)

    cap.release()

    # Show similarity chart
    if similarity_scores:
        st.subheader("📈 Similarity Score Over Frames")
        scores_array = np.array(similarity_scores)
        st.line_chart({
            "Similarity Score": scores_array[:, 1]
        })

    # Display accuracy
    accuracy = matched_count / total_frames * 100 if total_frames else 0
    st.success(f"✅ Matches Found: {matched_count}/{total_frames} frames ({accuracy:.2f}%)")

    # Save matched frames to video
    if matched_frames:
        height, width, _ = matched_frames[0].shape
        video_out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for frame in matched_frames:
            video_out.write(frame)
        video_out.release()

        st.subheader("▶️ Final Output Video with Matched Frames")
        with open(output_video, "rb") as f:
            st.video(f.read())
    else:
        st.warning("⚠️ No matched frames found.")

else:
    st.info("👆 Please upload both target image and video to begin.")
