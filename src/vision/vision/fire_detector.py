"""Real-time fire and smoke detection using YOLO with approximate distance estimation.

Controls:
- c: calibrate focal length using the largest visible fire
- r: reset distance calibration
- q: quit

Notes:
- Use a custom YOLO model trained for fire/smoke detection.
- This version is intended for webcam input.
- Distance estimation is approximate because fire does not have a fixed physical size.
"""

import cv2
import math
import os
from ultralytics import YOLO


# Input source.
# 0 for laptop webcam.
INPUT_SOURCE = 0

# Trained model from local run.
FIRE_MODEL_PATH = "runs/detect/train/weights/best.pt"

# YOLO inference parameters.
YOLO_CONFIDENCE = 0.45
YOLO_IMAGE_SIZE = 640

# Approximate real fire width in centimeters.
KNOWN_FIRE_WIDTH_CM = 30.0

# Stand at this distance during calibration and press `c`.
CALIBRATION_DISTANCE_CM = 100.0

# Fire/smoke class names expected from the custom model.
FIRE_CLASS_NAMES = ("fire", "flame")
SMOKE_CLASS_NAMES = ("smoke",)
VALID_CLASS_NAMES = FIRE_CLASS_NAMES + SMOKE_CLASS_NAMES

# Set True when using your trained fire/smoke model.
ENABLE_CLASS_FILTER = True

# Tracking parameters.
TRACK_MAX_MISSES = 15
TRACK_MAX_MATCH_DISTANCE = 120
BBOX_EMA_ALPHA = 0.35
CONFIDENCE_EMA_ALPHA = 0.30
DISTANCE_EMA_ALPHA = 0.25
MIN_IOU_FOR_MATCH = 0.05

# Temporal confirmation.
# Fire/smoke must be detected for several frames before being confirmed.
CONFIRMATION_FRAMES = 5

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX


def focal_length_finder(measured_distance_cm, real_width_cm, width_in_image_px):
    return (width_in_image_px * measured_distance_cm) / real_width_cm


def distance_finder(focal_length_px, real_width_cm, width_in_frame_px):
    return (real_width_cm * focal_length_px) / width_in_frame_px


def is_image_file(source):
    if not isinstance(source, str):
        return False

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return source.lower().endswith(image_extensions)


def clamp(value, low, high):
    return max(low, min(value, high))


def bbox_center(box):
    x, y, w, h = box
    return (x + (w / 2.0), y + (h / 2.0))


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = max(1, area_a + area_b - inter_area)
    return inter_area / union_area


def smooth_value(prev_value, measured_value, alpha):
    if prev_value is None:
        return measured_value
    return (alpha * measured_value) + ((1.0 - alpha) * prev_value)


def smooth_distance(prev_distance, measured_distance, alpha=DISTANCE_EMA_ALPHA):
    if prev_distance is None:
        return measured_distance
    return (alpha * measured_distance) + ((1.0 - alpha) * prev_distance)


def smooth_box(prev_box, measured_box, alpha=BBOX_EMA_ALPHA):
    if prev_box is None:
        return tuple(float(v) for v in measured_box)

    return tuple(
        (alpha * float(measured_box[i])) + ((1.0 - alpha) * float(prev_box[i]))
        for i in range(4)
    )


def int_box(box):
    x, y, w, h = box
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def xyxy_to_xywh(x1, y1, x2, y2):
    x = int(round(x1))
    y = int(round(y1))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))
    return (x, y, w, h)


def normalize_class_name(class_name):
    return str(class_name).strip().lower()


def detect_fire_objects(frame, model):
    detections = []

    results = model.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False,
    )

    if len(results) == 0:
        return detections

    result = results[0]
    names = result.names

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = normalize_class_name(names[class_id])

        if ENABLE_CLASS_FILTER and class_name not in VALID_CLASS_NAMES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = xyxy_to_xywh(x1, y1, x2, y2)

        detections.append(
            {
                "bbox": bbox,
                "class_name": class_name,
                "confidence": confidence,
            }
        )

    return detections


def update_tracks(detections, tracks, next_track_id):
    """Update track states and return active tracks as (track_id, track)."""
    used_tracks = set()

    detections = sorted(
        detections,
        key=lambda item: item["bbox"][2] * item["bbox"][3],
        reverse=True,
    )

    for detection in detections:
        box = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        center = bbox_center(box)
        _, _, w, h = box

        best_id = None
        best_score = float("inf")

        for track_id, track in tracks.items():
            if track_id in used_tracks:
                continue

            if track["class_name"] != class_name:
                continue

            prev_box = track["bbox"]
            iou = bbox_iou(tuple(box), prev_box)

            misses = track["misses"]
            adaptive_gate = max(TRACK_MAX_MATCH_DISTANCE, 1.2 * max(w, h))
            adaptive_gate *= (1.0 + 0.15 * misses)

            dist = euclidean_distance(center, track["center"])

            if dist > adaptive_gate and iou < MIN_IOU_FOR_MATCH:
                continue

            score = (1.0 - iou) + (dist / max(1.0, adaptive_gate))
            if score < best_score:
                best_score = score
                best_id = track_id

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1

            smoothed_box = smooth_box(None, box)

            tracks[best_id] = {
                "center": bbox_center(smoothed_box),
                "bbox": smoothed_box,
                "class_name": class_name,
                "misses": 0,
                "hit_count": 1,
                "confirmed": CONFIRMATION_FRAMES <= 1,
                "smoothed_confidence": confidence,
                "smoothed_distance": None,
            }
        else:
            smoothed_box = smooth_box(tracks[best_id]["bbox"], box)
            smoothed_confidence = smooth_value(
                tracks[best_id]["smoothed_confidence"],
                confidence,
                CONFIDENCE_EMA_ALPHA,
            )

            tracks[best_id]["center"] = bbox_center(smoothed_box)
            tracks[best_id]["bbox"] = smoothed_box
            tracks[best_id]["misses"] = 0
            tracks[best_id]["hit_count"] += 1
            tracks[best_id]["smoothed_confidence"] = smoothed_confidence

            if tracks[best_id]["hit_count"] >= CONFIRMATION_FRAMES:
                tracks[best_id]["confirmed"] = True

        used_tracks.add(best_id)

    stale_ids = []
    for track_id, track in tracks.items():
        if track_id not in used_tracks:
            track["misses"] += 1

            if track["misses"] > TRACK_MAX_MISSES:
                stale_ids.append(track_id)

    for track_id in stale_ids:
        tracks.pop(track_id, None)

    active_tracks = [
        (track_id, track)
        for track_id, track in tracks.items()
        if track["misses"] <= TRACK_MAX_MISSES
    ]

    active_tracks.sort(key=lambda item: item[0])
    return active_tracks, tracks, next_track_id


def update_fire_distance(track, focal_length_found, box_width_px):
    if focal_length_found is None:
        return None

    if box_width_px <= 0:
        return track["smoothed_distance"]

    distance_cm = distance_finder(
        focal_length_found,
        KNOWN_FIRE_WIDTH_CM,
        box_width_px,
    )

    smoothed_distance = smooth_distance(
        track["smoothed_distance"],
        distance_cm,
    )

    track["smoothed_distance"] = smoothed_distance
    return smoothed_distance


def draw_tracks(frame, active_tracks, focal_length_found):
    fire_confirmed = False
    smoke_confirmed = False

    for track_id, track in active_tracks:
        x, y, w, h = int_box(track["bbox"])

        if w <= 0 or h <= 0:
            continue

        class_name = track["class_name"]
        confidence = track["smoothed_confidence"]
        confirmed = track["confirmed"]

        if class_name in FIRE_CLASS_NAMES:
            color = RED if confirmed else ORANGE
        else:
            color = YELLOW if confirmed else GREEN

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if confirmed:
            status = "CONFIRMED"
        else:
            status = "checking"

        distance_text = ""

        if class_name in FIRE_CLASS_NAMES:
            if track["misses"] == 0:
                smoothed_distance = update_fire_distance(
                    track,
                    focal_length_found,
                    w,
                )
            else:
                smoothed_distance = track["smoothed_distance"]

            if smoothed_distance is not None:
                distance_text = f" | {smoothed_distance:.1f} CM"

        label = (
            f"ID {track_id} | {class_name} | "
            f"{confidence:.2f}{distance_text} | {status}"
        )

        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            FONT,
            0.55,
            color,
            2,
        )

        if confirmed and class_name in FIRE_CLASS_NAMES:
            fire_confirmed = True

        if confirmed and class_name in SMOKE_CLASS_NAMES:
            smoke_confirmed = True

    return fire_confirmed, smoke_confirmed


def draw_status_bar(frame, fire_confirmed, smoke_confirmed, focal_length_found):
    cv2.line(frame, (20, 30), (720, 30), RED, 30)
    cv2.line(frame, (20, 30), (720, 30), BLACK, 26)

    if focal_length_found is None:
        calibration_text = "Press c to calibrate distance"
    else:
        calibration_text = "Distance calibrated | r reset"

    if fire_confirmed:
        status_text = f"FIRE CONFIRMED | {calibration_text} | q quit"
        status_color = RED
    elif smoke_confirmed:
        status_text = f"SMOKE CONFIRMED | {calibration_text} | q quit"
        status_color = YELLOW
    else:
        status_text = f"Monitoring fire/smoke | {calibration_text} | q quit"
        status_color = GREEN

    cv2.putText(
        frame,
        status_text,
        (30, 36),
        FONT,
        0.48,
        status_color,
        2,
    )


def process_frame(frame, model, tracks, next_track_id, focal_length_found):
    detections = detect_fire_objects(frame, model)
    active_tracks, tracks, next_track_id = update_tracks(
        detections,
        tracks,
        next_track_id,
    )

    fire_confirmed, smoke_confirmed = draw_tracks(
        frame,
        active_tracks,
        focal_length_found,
    )

    draw_status_bar(
        frame,
        fire_confirmed,
        smoke_confirmed,
        focal_length_found,
    )

    return frame, tracks, next_track_id, fire_confirmed, smoke_confirmed


def calibrate_distance_from_largest_fire(tracks):
    fire_tracks = [
        track
        for track in tracks.values()
        if track["class_name"] in FIRE_CLASS_NAMES
        and track["misses"] == 0
    ]

    if len(fire_tracks) == 0:
        print("Calibration failed: no visible fire detected.")
        return None

    largest_track = max(
        fire_tracks,
        key=lambda track: track["bbox"][2] * track["bbox"][3],
    )

    _, _, fire_width_px, _ = int_box(largest_track["bbox"])

    if fire_width_px <= 0:
        print("Calibration failed: invalid fire width.")
        return None

    focal_length_found = focal_length_finder(
        CALIBRATION_DISTANCE_CM,
        KNOWN_FIRE_WIDTH_CM,
        fire_width_px,
    )

    print(f"Calibrated focal length: {focal_length_found:.2f} px")
    print(f"Known fire width: {KNOWN_FIRE_WIDTH_CM:.1f} cm")
    print(f"Calibration distance: {CALIBRATION_DISTANCE_CM:.1f} cm")

    return focal_length_found


def run_video_source(model):
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {INPUT_SOURCE}")

    tracks = {}
    next_track_id = 1
    focal_length_found = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame, tracks, next_track_id, fire_confirmed, smoke_confirmed = process_frame(
                frame,
                model,
                tracks,
                next_track_id,
                focal_length_found,
            )

            cv2.imshow("Fire and Smoke Distance Estimator", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                focal_length_candidate = calibrate_distance_from_largest_fire(tracks)

                if focal_length_candidate is not None:
                    focal_length_found = focal_length_candidate

                    for track in tracks.values():
                        track["smoothed_distance"] = None

            if key == ord("r"):
                focal_length_found = None

                for track in tracks.values():
                    track["smoothed_distance"] = None

                print("Distance calibration reset.")

            if key == ord("q"):
                break
    finally:
        cap.release()


def main():
    if not os.path.exists(FIRE_MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {FIRE_MODEL_PATH}\n"
            "Please set FIRE_MODEL_PATH to your custom fire/smoke YOLO model."
        )

    model = YOLO(FIRE_MODEL_PATH)

    try:
        run_video_source(model)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()