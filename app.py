import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Video Detection", layout="wide")
st.title("Deteksi Penggunaan Helm Oleh Pengemudi")

# =========================
# Sidebar - Upload Model
# =========================
st.sidebar.header("Upload YOLO Model (.pt)")
model_file = st.sidebar.file_uploader(
    "Upload model YOLO (.pt)", type=["pt"]
)

# =========================
# Sidebar - Pengaturan
# =========================
st.sidebar.header("⚙️ Pengaturan")

mode = st.sidebar.radio(
    "Sumber Video",
    ["Upload Video", "URL Stream"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
)

IoU = st.sidebar.slider(
    "IoU Threshold", 0.1, 1.0, 0.7, 0.05
)

# =========================
# URL Stream Input
# =========================
stream_url = ""
if mode == "URL Stream":
    st.sidebar.header("Video Source URL")
    stream_url = st.sidebar.text_input(
        "Masukkan URL stream (RTSP / HTTP)",
        placeholder="rtsp://... atau http://..."
    )

# =========================
# 1️⃣ MODE: UPLOAD VIDEO
# =========================
if mode == "Upload Video":

    video_file = st.sidebar.file_uploader(
        "Upload Video", type=["mp4", "mov"]
    )

    run_button = st.sidebar.button("🚀 Run Detection")

    if run_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name

        model = YOLO(model_path)
        names = model.names

        CLASS_COLORS = {
            0: (0, 255, 0),
            1: (0, 0, 255),
            2: (255, 0, 0)
        }
        
        if model_file is None:
            st.error("❌ Upload model YOLO terlebih dahulu")
            st.stop()

        if video_file is None:
            st.warning("⚠️ Upload video")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        processed = 0

        st.info("🔍 Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence, iou=IoU, verbose=False)
            annotated = frame.copy()

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = f"{names[cls_id]} {conf:.2f}"
                    color = CLASS_COLORS.get(cls_id, (255,255,255))

                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(
                        annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
                    )

            out.write(annotated)
            processed += 1
            progress.progress(processed / frame_count)

        cap.release()
        out.release()

        st.success("🎉 Detection finished!")

        st.download_button(
            "⬇️ Download Result Video",
            data=open(output_path, "rb"),
            file_name="hasil.mp4",
            mime="video/mp4"
        )

# =========================
# 2️⃣ MODE: URL STREAM (REAL-TIME)
# =========================
if mode == "URL Stream":
    run_button = st.sidebar.button("🚀 Run Detection")

    if run_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name

        model = YOLO(model_path)
        names = model.names

        CLASS_COLORS = {
            0: (0, 255, 0),
            1: (0, 0, 255),
            2: (255, 0, 0)
        }        

        if stream_url.strip() == "":
            st.error("❌ Masukkan URL stream")
            st.stop()
            
        if model_file is None:
            st.error("❌ Upload model YOLO terlebih dahulu")
            st.stop()

        cap = cv2.VideoCapture(stream_url)
        frame_placeholder = st.empty()

        st.info("📡 Menghubungkan ke stream...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Stream terputus")
                break

            # OPTIONAL: resize untuk FPS lebih tinggi
            # frame = cv2.resize(frame, (640, 640))

            results = model.predict(frame, conf=confidence, iou=IoU, verbose=False)
            annotated = frame.copy()

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = f"{names[cls_id]} {conf:.2f}"
                    color = CLASS_COLORS.get(cls_id, (255,255,255))

                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(
                        annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
                    )

            frame_placeholder.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()