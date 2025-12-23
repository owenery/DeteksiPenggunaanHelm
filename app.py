import streamlit as st
import tempfile
import cv2
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

    names = model.names

    CLASS_COLORS = {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (255, 0, 0)
    }

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
        annotated_frame = frame.copy()

        if results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = f"{names[cls_id]} {conf:.2f}"
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                # Bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness=2
                )

                # Label background
                (w, h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - h - 8),
                    (x1 + w + 4, y1),
                    color,
                    -1
                )

                # Label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        out.write(annotated_frame)

        processed += 1
        progress.progress(processed / frame_count)

    cap.release()
    out.release()

    st.success("🎉 Detection finished!")

    st.download_button(
        label="⬇️ Download Result Video",
        data=open(output_path, "rb"),
        file_name="hasil.mp4",
        mime="video/mp4"
    )