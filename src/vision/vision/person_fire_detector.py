"""Real-time fire and human detection using webcam.

Controls:
- q: quit

Notes:
- Fire is detected using a custom YOLO model.
- Human/person is detected using pretrained YOLO11.
- Detections are merged into one tracking pipeline.
"""

import cv2
import math
import os
from ultralytics import YOLO


# Input source.
INPUT_SOURCE = 0

# Model paths.
VISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRE_MODEL_PATH = os.path.join(
    VISION_ROOT, "runs", "detect", "train", "weights", "best.pt"
)
PERSON_MODEL_PATH = "yolo11n.pt"

# YOLO inference parameters.
YOLO_CONFIDENCE_FIRE = 0.25
YOLO_CONFIDENCE_PERSON = 0.25
YOLO_IOU = 0.50
YOLO_IMAGE_SIZE = 640

# Class names.
FIRE_CLASS_NAMES = ("fire", "flame")
PERSON_CLASS_NAMES = ("person",)
VALID_CLASS_NAMES = FIRE_CLASS_NAMES + PERSON_CLASS_NAMES

# Tracking parameters.
TRACK_MAX_MISSES = 15
TRACK_MAX_MATCH_DISTANCE = 120
BBOX_EMA_ALPHA = 0.35
CONFIDENCE_EMA_ALPHA = 0.30
MIN_IOU_FOR_MATCH = 0.05
# Fire-person interaction.
INTERACTION_IOMIN_THRESHOLD = 0.15

# Temporal confirmation.
FIRE_CONFIRMATION_FRAMES = 5
PERSON_CONFIRMATION_FRAMES = 3

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX


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

def bbox_intersection_area(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return inter_w * inter_h


def bbox_iomin(box_a, box_b):
    """Intersection over the smaller box area. Suitable for objects of different sizes."""
    inter = bbox_intersection_area(box_a, box_b)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    min_area = max(1.0, min(area_a, area_b))
    return inter / min_area

def smooth_value(prev_value, measured_value, alpha):
    if prev_value is None:
        return measured_value
    return (alpha * measured_value) + ((1.0 - alpha) * prev_value)


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


def get_confirmation_frames(class_name):
    if class_name in FIRE_CLASS_NAMES:
        return FIRE_CONFIRMATION_FRAMES

    if class_name in PERSON_CLASS_NAMES:
        return PERSON_CONFIRMATION_FRAMES

    return PERSON_CONFIRMATION_FRAMES


def detect_objects(frame, model, valid_class_names, confidence):
    detections = []

    results = model.predict(
        frame,
        conf=confidence,
        iou=YOLO_IOU,
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
        confidence_score = float(box.conf[0])
        class_name = normalize_class_name(names[class_id])

        if class_name not in valid_class_names:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = xyxy_to_xywh(x1, y1, x2, y2)

        detections.append(
            {
                "bbox": bbox,
                "class_name": class_name,
                "confidence": confidence_score,
            }
        )

    return detections


def detect_fire_and_person(frame, fire_model, person_model):
    detections = []

    fire_detections = detect_objects(
        frame,
        fire_model,
        FIRE_CLASS_NAMES,
        YOLO_CONFIDENCE_FIRE,
    )

    person_detections = detect_objects(
        frame,
        person_model,
        PERSON_CLASS_NAMES,
        YOLO_CONFIDENCE_PERSON,
    )

    detections.extend(fire_detections)
    detections.extend(person_detections)

    return detections

def find_fire_person_interactions(active_tracks, threshold=INTERACTION_IOMIN_THRESHOLD):
    """Return list of (fire_id, person_id, overlap) yang beririsan."""
    fires = [
        (tid, t) for tid, t in active_tracks
        if t["confirmed"] and t["class_name"] in FIRE_CLASS_NAMES
    ]
    persons = [
        (tid, t) for tid, t in active_tracks
        if t["confirmed"] and t["class_name"] in PERSON_CLASS_NAMES
    ]

    interactions = []
    for fire_id, fire_track in fires:
        for person_id, person_track in persons:
            overlap = bbox_iomin(fire_track["bbox"], person_track["bbox"])
            if overlap >= threshold:
                interactions.append((fire_id, person_id, overlap))

    return interactions

def draw_interactions(frame, active_tracks, interactions):
    track_map = dict(active_tracks)

    for fire_id, person_id, overlap in interactions:
        fire_box = int_box(track_map[fire_id]["bbox"])
        person_box = int_box(track_map[person_id]["bbox"])

        # Hitung kotak irisan untuk di-highlight.
        fx, fy, fw, fh = fire_box
        px, py, pw, ph = person_box

        ix1 = max(fx, px)
        iy1 = max(fy, py)
        ix2 = min(fx + fw, px + pw)
        iy2 = min(fy + fh, py + ph)

        if ix2 > ix1 and iy2 > iy1:
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), ORANGE, 3)
            cv2.putText(
                frame,
                f"INTERACT F{fire_id}-P{person_id} {overlap:.2f}",
                (ix1, max(20, iy1 - 10)),
                FONT,
                0.55,
                ORANGE,
                2,
            )

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
                "confirmed": get_confirmation_frames(class_name) <= 1,
                "smoothed_confidence": confidence,
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

            if tracks[best_id]["hit_count"] >= get_confirmation_frames(class_name):
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


def get_track_color(class_name, confirmed):
    if class_name in FIRE_CLASS_NAMES:
        return RED if confirmed else ORANGE

    if class_name in PERSON_CLASS_NAMES:
        return GREEN if confirmed else BLUE

    return GREEN


def draw_tracks(frame, active_tracks):
    fire_count = 0
    person_count = 0

    for track_id, track in active_tracks:
        x, y, w, h = int_box(track["bbox"])

        if w <= 0 or h <= 0:
            continue

        class_name = track["class_name"]
        confidence = track["smoothed_confidence"]
        confirmed = track["confirmed"]

        color = get_track_color(class_name, confirmed)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        if confirmed:
            status = "CONFIRMED"
        else:
            status = "checking"

        label = f"ID {track_id} | {class_name} | {confidence:.2f} | {status}"

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
            fire_count += 1

        if confirmed and class_name in PERSON_CLASS_NAMES:
            person_count += 1

    return fire_count, person_count


def draw_status_bar(frame, fire_count, person_count, interaction_count=0):
    cv2.line(frame, (20, 30), (650, 30), RED, 30)
    cv2.line(frame, (20, 30), (650, 30), BLACK, 26)

    status_text = (
        f"Fire: {fire_count} | "
        f"Person: {person_count} | "
        f"Interact: {interaction_count} | "
        "q quit"
    )

    cv2.putText(frame, status_text, (30, 36), FONT, 0.55, GREEN, 2)


def process_frame(frame, fire_model, person_model, tracks, next_track_id):
    detections = detect_fire_and_person(frame, fire_model, person_model)

    active_tracks, tracks, next_track_id = update_tracks(
        detections, tracks, next_track_id,
    )

    fire_count, person_count = draw_tracks(frame, active_tracks)

    interactions = find_fire_person_interactions(active_tracks)
    draw_interactions(frame, active_tracks, interactions)

    draw_status_bar(frame, fire_count, person_count, len(interactions))

    return frame, tracks, next_track_id, fire_count, person_count


def open_video_capture(source):
    if isinstance(source, int):
        backend_candidates = []

        if os.name == "nt":
            backend_candidates.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])

        backend_candidates.append(None)

        for backend in backend_candidates:
            if backend is None:
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source, backend)

            if not cap.isOpened():
                cap.release()
                continue

            ok, _ = cap.read()
            if ok:
                return cap

            cap.release()

        raise RuntimeError(
            "Cannot open webcam source. Tried DirectShow/MSMF/default backends."
        )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {source}")

    return cap


def main():
    if not os.path.exists(FIRE_MODEL_PATH):
        raise RuntimeError(
            f"Fire model not found: {FIRE_MODEL_PATH}\n"
            "Please put your trained fire model at models/fire_best.pt"
        )

    fire_model = YOLO(FIRE_MODEL_PATH)
    person_model = YOLO(PERSON_MODEL_PATH)

    cap = open_video_capture(INPUT_SOURCE)

    tracks = {}
    next_track_id = 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame, tracks, next_track_id, fire_count, person_count = process_frame(
                frame, fire_model, person_model, tracks, next_track_id)

            cv2.imshow("Fire and Human Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()