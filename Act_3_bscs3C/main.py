import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection, Counting & Alerts")

# Select objects to detect
all_classes = model.names
selected_classes = st.multiselect(
    "Select objects to detect:",
    options=list(all_classes.values()),
    default=["person"]
)

# Alert threshold
threshold = st.slider("Alert when object count exceeds:", 1, 10, 3)

# Map class names to IDs
selected_ids = [k for k, v in all_classes.items() if v in selected_classes]

# Frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    boxes = results[0].boxes
    counts = {}

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])

            if cls_id in selected_ids:
                class_name = model.names[cls_id]
                counts[class_name] = counts.get(class_name, 0) + 1

    # Draw annotations
    annotated_frame = results[0].plot()

    # Overlay counts
    y_offset = 30
    for obj, count in counts.items():
        text = f"{obj}: {count}"
        cv2.putText(
            annotated_frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y_offset += 30

    # Alert condition
    for obj, count in counts.items():
        if count >= threshold:
            cv2.putText(
                annotated_frame,
                f"ALERT: Too many {obj}s!",
                (10, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)