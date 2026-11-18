import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI Parking Detection System",
    page_icon="🚗",
    layout="wide",
)

st.title("AI-Based Vehicle Parking Slot Detection")
st.caption(
    "Upload a suitable parking image or capture one. "
    "The system detects empty and occupied parking slots and assigns "
    "temporary slot IDs automatically."
)

# Model path resolution: works both locally on Windows and on Streamlit Cloud Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "weights.pt")

if os.path.isfile(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
elif os.path.isfile(r"F:\archive\weights.pt"):
    MODEL_PATH = r"F:\archive\weights.pt"
else:
    MODEL_PATH = "weights.pt"
CONFIDENCE_THRESHOLD = 0.30


# --------------------------------------------------
# LOAD TRAINED RF-DETR MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Ensure weights.pt is present in the project directory."
        )
    return RFDETRSmall(pretrain_weights=MODEL_PATH)


try:
    with st.spinner("Loading RF-DETR model..."):
        model = load_model()
    st.success("RF-DETR model loaded successfully.")
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


# Dataset class mapping:
# 0 = spaces (parent/group class; ignore)
# 1 = space-empty
# 2 = space-occupied
EMPTY_CLASS_ID = 1
OCCUPIED_CLASS_ID = 2


# --------------------------------------------------
# IOU + NON-MAXIMUM SUPPRESSION
# --------------------------------------------------
def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection
    return intersection / (union + 1e-6)


def remove_duplicate_detections(boxes, confidences, class_ids, iou_threshold=0.50):
    if len(boxes) == 0:
        return sv.Detections.empty()

    keep = []

    # Suppress duplicates within each class. Different classes are kept
    # separate because an empty slot and an occupied slot are distinct labels.
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        indices = indices[np.argsort(confidences[indices])[::-1]]

        while len(indices) > 0:
            current = int(indices[0])
            keep.append(current)
            remaining = indices[1:]

            indices = np.array(
                [
                    int(index)
                    for index in remaining
                    if calculate_iou(boxes[current], boxes[int(index)]) < iou_threshold
                ],
                dtype=int,
            )

    keep = np.array(sorted(keep), dtype=int)
    return sv.Detections(
        xyxy=boxes[keep].astype(np.float32),
        confidence=confidences[keep].astype(np.float32),
        class_id=class_ids[keep].astype(int),
    )


# --------------------------------------------------
# TILED INFERENCE
# --------------------------------------------------
def tiled_detection(image, threshold=CONFIDENCE_THRESHOLD):
    """Run RF-DETR on overlapping image tiles and map boxes to full-image coordinates."""
    width, height = image.size

    tile_w = max(1, int(width * 0.35))
    tile_h = max(1, int(height * 0.35))

    all_boxes = []
    all_confidences = []
    all_class_ids = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for row in range(3):
            for col in range(3):
                x1 = int(col * (width - tile_w) / 2)
                y1 = int(row * (height - tile_h) / 2)
                x2 = min(x1 + tile_w, width)
                y2 = min(y1 + tile_h, height)

                if x2 <= x1 or y2 <= y1:
                    continue

                tile = image.crop((x1, y1, x2, y2))
                tile_path = os.path.join(temp_dir, f"tile_{row}_{col}.jpg")
                tile.save(tile_path, quality=95)

                predictions = model.predict(tile_path, threshold=threshold)

                if predictions is None or len(predictions.xyxy) == 0:
                    continue

                for box, confidence, class_id in zip(
                    predictions.xyxy,
                    predictions.confidence,
                    predictions.class_id,
                ):
                    class_id = int(class_id)

                    # Ignore parent/group label and any unexpected class.
                    if class_id not in (EMPTY_CLASS_ID, OCCUPIED_CLASS_ID):
                        continue

                    full_box = np.asarray(box, dtype=np.float32).copy()
                    full_box[[0, 2]] += x1
                    full_box[[1, 3]] += y1

                    # Clip boxes to the original image.
                    full_box[[0, 2]] = np.clip(full_box[[0, 2]], 0, width)
                    full_box[[1, 3]] = np.clip(full_box[[1, 3]], 0, height)

                    all_boxes.append(full_box)
                    all_confidences.append(float(confidence))
                    all_class_ids.append(class_id)

    if not all_boxes:
        return sv.Detections.empty()

    boxes = np.asarray(all_boxes, dtype=np.float32).reshape(-1, 4)
    confidences = np.asarray(all_confidences, dtype=np.float32)
    class_ids = np.asarray(all_class_ids, dtype=int)

    return remove_duplicate_detections(boxes, confidences, class_ids)


# --------------------------------------------------
# AUTOMATIC SLOT NUMBERING
# --------------------------------------------------
def assign_slot_numbers(detections, starting_number=1):
    """
    Assign temporary IDs based on image position:
    top-to-bottom rows, then left-to-right within each row.

    These IDs are not real-world plot numbers unless a registered site map
    is used to match them.
    """
    if len(detections) == 0:
        return []

    boxes = detections.xyxy
    centers = np.column_stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        )
    )
    heights = boxes[:, 3] - boxes[:, 1]
    positive_heights = heights[heights > 0]
    median_height = float(np.median(positive_heights)) if len(positive_heights) else 20.0
    row_tolerance = max(12.0, median_height * 0.55)

    # Sort by vertical centre first.
    indices = sorted(range(len(centers)), key=lambda i: (centers[i, 1], centers[i, 0]))

    rows = []
    for index in indices:
        cy = centers[index, 1]

        best_row = None
        best_distance = None
        for row in rows:
            row_center = float(np.mean([centers[item, 1] for item in row]))
            distance = abs(cy - row_center)
            if distance <= row_tolerance and (best_distance is None or distance < best_distance):
                best_row = row
                best_distance = distance

        if best_row is None:
            rows.append([index])
        else:
            best_row.append(index)

    ordered_indices = []
    rows.sort(key=lambda row: float(np.mean([centers[item, 1] for item in row])))
    for row in rows:
        ordered_indices.extend(sorted(row, key=lambda i: centers[i, 0]))

    slot_numbers = [None] * len(detections)
    for offset, detection_index in enumerate(ordered_indices):
        slot_numbers[detection_index] = int(starting_number) + offset

    return slot_numbers


def make_slot_records(detections, slot_numbers):
    records = []
    for index, (class_id, confidence, box) in enumerate(
        zip(detections.class_id, detections.confidence, detections.xyxy)
    ):
        if int(class_id) == EMPTY_CLASS_ID:
            status = "Empty"
        elif int(class_id) == OCCUPIED_CLASS_ID:
            status = "Occupied"
        else:
            continue

        records.append(
            {
                "Slot ID": f"Slot {slot_numbers[index]}",
                "Status": status,
                "Confidence": round(float(confidence), 3),
                "X1": int(box[0]),
                "Y1": int(box[1]),
                "X2": int(box[2]),
                "Y2": int(box[3]),
            }
        )
    return records


def annotate_parking_image(image_np, detections, slot_numbers):
    annotated = image_np.copy()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels = []
    for class_id, confidence, slot_number in zip(
        detections.class_id, detections.confidence, slot_numbers
    ):
        status = "EMPTY" if int(class_id) == EMPTY_CLASS_ID else "OCCUPIED"
        labels.append(f"Slot {slot_number} | {status} | {float(confidence):.2f}")

    if len(detections) > 0:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )
    return annotated


# --------------------------------------------------
# IMAGE / CAMERA INPUT
# --------------------------------------------------
st.subheader("Parking Image Input")
uploaded_file = st.file_uploader(
    "Upload any suitable parking image",
    type=["jpg", "jpeg", "png"],
)
camera_file = st.camera_input("Or capture a parking image")

image_file = uploaded_file if uploaded_file is not None else camera_file

col_a, col_b = st.columns(2)
with col_a:
    parking_location = st.text_input(
        "Parking location (optional)",
        placeholder="Example: College Main Parking Area",
    )
with col_b:
    starting_number = st.number_input(
        "Temporary slot numbering starts from",
        min_value=1,
        max_value=999999,
        value=1,
        step=1,
    )

threshold = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=CONFIDENCE_THRESHOLD,
    step=0.05,
)


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------
if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        image_np = np.asarray(image)

        st.subheader("Input Image")
        st.image(image_np, use_container_width=True)

        with st.spinner("Analysing parking image with RF-DETR..."):
            detections = tiled_detection(image, threshold=threshold)
            slot_numbers = assign_slot_numbers(detections, int(starting_number))
            slot_records = make_slot_records(detections, slot_numbers)
            annotated_image = annotate_parking_image(
                image_np, detections, slot_numbers
            )

        empty_count = int(np.sum(detections.class_id == EMPTY_CLASS_ID))
        occupied_count = int(np.sum(detections.class_id == OCCUPIED_CLASS_ID))
        total_slots = empty_count + occupied_count
        availability = (empty_count / total_slots * 100) if total_slots else 0.0

        st.subheader("Parking Analysis Summary")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Total Slots Detected", total_slots)
        metric2.metric("Empty Slots", empty_count)
        metric3.metric("Occupied Slots", occupied_count)
        metric4.metric("Availability", f"{availability:.1f}%")

        st.subheader("Automatically Numbered Detection")
        st.image(annotated_image, use_container_width=True)

        st.subheader("Plot-wise Availability")
        if slot_records:
            slot_df = pd.DataFrame(slot_records)
            st.dataframe(slot_df, use_container_width=True, hide_index=True)

            empty_slots = [
                record["Slot ID"]
                for record in slot_records
                if record["Status"] == "Empty"
            ]
            if empty_slots:
                notification_text = (
                    f"Parking update for {parking_location or 'selected parking area'}: "
                    f"{', '.join(empty_slots)} are currently detected as empty."
                )
                st.success(notification_text)
                st.download_button(
                    "Download availability message",
                    data=notification_text,
                    file_name="parking_availability_message.txt",
                    mime="text/plain",
                )
            else:
                st.warning("No empty parking slots were detected in this image.")

            now = datetime.now()
            report_df = pd.DataFrame(
                [
                    {
                        "Date": now.strftime("%d-%m-%Y"),
                        "Day": now.strftime("%A"),
                        "Time": now.strftime("%H:%M:%S"),
                        "Location": parking_location,
                        "Image": getattr(image_file, "name", "camera_capture"),
                        "Total Slots": total_slots,
                        "Empty Slots": empty_count,
                        "Occupied Slots": occupied_count,
                        "Availability (%)": round(availability, 2),
                    }
                ]
            )

            st.subheader("Monitoring Information")
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            csv_bytes = slot_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download plot-wise slot report (CSV)",
                data=csv_bytes,
                file_name="parking_slot_report.csv",
                mime="text/csv",
            )

            summary_csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download parking summary (CSV)",
                data=summary_csv,
                file_name="parking_summary.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "No parking slots were detected. Try a clearer image showing "
                "the parking spaces, or adjust the confidence threshold."
            )

        st.info(
            "Note: Slot 1, Slot 2, etc. are temporary IDs assigned by image position. "
            "They are not actual plot numbers such as Plot 78. Real plot numbers "
            "require a registered parking layout/map."
        )

    except Exception as exc:
        st.error(f"Image processing failed: {exc}")


# --------------------------------------------------
# DRONE VIDEO INPUT (FRAME-BY-FRAME DEMO)
# --------------------------------------------------
st.divider()
st.subheader("Drone Video Monitoring")
st.caption(
    "This processes sampled video frames. Stable plot identity across moving "
    "camera frames requires a calibrated site map or tracking/georeferencing."
)

video_file = st.file_uploader(
    "Upload drone video",
    type=["mp4", "avi", "mov"],
    key="drone_video",
)

if video_file is not None:
    st.video(video_file)

    if st.button("Start Video Detection"):
        video_path = None
        cap = None
        try:
            suffix = os.path.splitext(video_file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                temp_video.write(video_file.getbuffer())
                video_path = temp_video.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Could not open this video file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_interval = max(1, int(round(fps)))  # approximately one frame/second

                progress = st.progress(0.0)
                frame_display = st.empty()
                result_display = st.empty()
                frame_number = 0

                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    frame_number += 1
                    if frame_number % frame_interval == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_image = Image.fromarray(rgb_frame)

                        frame_detections = tiled_detection(
                            frame_image,
                            threshold=threshold,
                        )

                        frame_empty = int(
                            np.sum(frame_detections.class_id == EMPTY_CLASS_ID)
                        )
                        frame_occupied = int(
                            np.sum(frame_detections.class_id == OCCUPIED_CLASS_ID)
                        )
                        frame_ids = assign_slot_numbers(frame_detections, 1)
                        frame_annotated = annotate_parking_image(
                            rgb_frame,
                            frame_detections,
                            frame_ids,
                        )

                        frame_display.image(
                            frame_annotated,
                            caption=f"Processed frame {frame_number}",
                            use_container_width=True,
                        )
                        result_display.write(
                            f"Empty slots: {frame_empty} | "
                            f"Occupied slots: {frame_occupied} | "
                            f"Total slots: {frame_empty + frame_occupied}"
                        )

                    if total_frames > 0:
                        progress.progress(min(frame_number / total_frames, 1.0))

                st.success("Video frame analysis completed.")

        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
