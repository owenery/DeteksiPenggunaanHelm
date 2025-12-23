import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Video Detection", layout="wide")

st.title("Deteksi Penggunaan Helm Oleh Pengemudi")

# =========================
# Upload model
# =========================
st.sidebar.header("Upload YOLO Model (.pt)")
model_file = st.sidebar.file_uploader(
    "Upload model YOLO (.pt)", type=["pt"]
)

# =========================
# Upload video
# =========================
st.sidebar.header("Upload Video")
video_file = st.sidebar.file_uploader(
    "Upload video", type=["mp4", "mov"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

IoU = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.05
)

run_button = st.sidebar.button("🚀 Run Detection")

# =========================
# Main Logic
# =========================
if run_button:
    if model_file is None or video_file is None:
        st.error("❌ Please upload both model and video.")
        st.stop()

    # ---- Save model temporarily ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
        tmp_model.write(model_file.read())
        model_path = tmp_model.name

    # ---- Save video temporarily ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    st.success("✅ Files uploaded successfully")

    # ---- Load model ----
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- Output video ----
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    st.info("🔍 Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(frame, conf=confidence, iou=IoU, verbose=False)

        # Plot results
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        processed += 1
        progress.progress(processed / frame_count)

    cap.release()
    out.release()

    st.success("🎉 Detection finished!")

    # =========================
    # Show results
    # =========================
    st.video(output_path)

    st.download_button(
        label="⬇️ Download Result Video",
        data=open(output_path, "rb"),
        file_name="result_detection.mp4",
        mime="video/mp4"
    )