"""Real-time face distance estimation using laptop webcam.

Controls:
- c: calibrate focal length using the largest visible face
- r: reset calibration
- q: quit
"""

import cv2
import math
import mediapipe as mp


# Real face width in centimeters (average adult face width).
KNOWN_FACE_WIDTH_CM = 14.3

# Stand at this distance during calibration and press `c`.
CALIBRATION_DISTANCE_CM = 60.0

# Tracking parameters.
TRACK_MAX_MISSES = 30
TRACK_MAX_MATCH_DISTANCE = 140
DISTANCE_EMA_ALPHA = 0.25
BBOX_EMA_ALPHA = 0.35
MIN_IOU_FOR_MATCH = 0.02

# If only one face and one old track exist, keep the same ID more aggressively.
SINGLE_FACE_REUSE_MAX_MISSES = 60

# MediaPipe face detector parameters.
# model_selection = 0: short-range face detector.
# model_selection = 1: full-range face detector.
FACE_MODEL_SELECTION = 0
MIN_DETECTION_CONFIDENCE = 0.60

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX


def focal_length_finder(measured_distance_cm, real_width_cm, width_in_image_px):
    return (width_in_image_px * measured_distance_cm) / real_width_cm


def distance_finder(focal_length_px, real_width_cm, width_in_frame_px):
    return (real_width_cm * focal_length_px) / width_in_frame_px


def clamp(value, low, high):
    return max(low, min(value, high))


def mediapipe_box_to_pixel_box(relative_box, frame_width, frame_height):
    x = int(relative_box.xmin * frame_width)
    y = int(relative_box.ymin * frame_height)
    w = int(relative_box.width * frame_width)
    h = int(relative_box.height * frame_height)

    x = clamp(x, 0, frame_width - 1)
    y = clamp(y, 0, frame_height - 1)
    w = clamp(w, 1, frame_width - x)
    h = clamp(h, 1, frame_height - y)

    return (x, y, w, h)


def detect_faces(frame, detector):
    frame_height, frame_width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = detector.process(rgb)
    rgb.flags.writeable = True

    face_boxes = []

    if not results.detections:
        return face_boxes

    for detection in results.detections:
        relative_box = detection.location_data.relative_bounding_box
        box = mediapipe_box_to_pixel_box(
            relative_box,
            frame_width,
            frame_height,
        )
        face_boxes.append(box)

    return face_boxes


def face_center(face_box):
    x, y, w, h = face_box
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


def int_box(face_box):
    x, y, w, h = face_box
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def update_tracks(face_boxes, tracks, next_track_id):
    """Update track states and return active tracks as (track_id, track)."""
    used_tracks = set()

    only_one_face = len(face_boxes) == 1
    only_one_track = len(tracks) == 1

    for box in sorted(face_boxes, key=lambda f: f[2] * f[3], reverse=True):
        center = face_center(box)
        _, _, w, h = box

        best_id = None
        best_score = float("inf")

        for track_id, track in tracks.items():
            if track_id in used_tracks:
                continue

            prev_box = track["bbox"]
            iou = bbox_iou(tuple(box), prev_box)

            # Adaptive gating: allow wider movement for larger boxes and missed frames.
            misses = track["misses"]
            adaptive_gate = max(TRACK_MAX_MATCH_DISTANCE, 1.2 * max(w, h))
            adaptive_gate *= (1.0 + 0.20 * misses)

            dist = euclidean_distance(center, track["center"])

            # Strong single-face reuse rule.
            # This prevents ID growth when one user briefly disappears/reappears.
            if only_one_face and only_one_track and misses <= SINGLE_FACE_REUSE_MAX_MISSES:
                best_id = track_id
                break

            # Match is valid if centroid is close enough OR boxes overlap enough.
            if dist > adaptive_gate and iou < MIN_IOU_FOR_MATCH:
                continue

            # Lower score is better: prioritize overlap, then center distance.
            score = (1.0 - iou) + (dist / max(1.0, adaptive_gate))
            if score < best_score:
                best_score = score
                best_id = track_id

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1
            smoothed_box = smooth_box(None, box)

            tracks[best_id] = {
                "center": face_center(smoothed_box),
                "bbox": smoothed_box,
                "misses": 0,
                "smoothed_distance": None,
            }
        else:
            smoothed_box = smooth_box(tracks[best_id]["bbox"], box)

            tracks[best_id]["center"] = face_center(smoothed_box)
            tracks[best_id]["bbox"] = smoothed_box
            tracks[best_id]["misses"] = 0

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


def main():
    mp_face_detection = mp.solutions.face_detection

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (camera index 0).")

    focal_length_found = None
    tracks = {}
    next_track_id = 1

    try:
        with mp_face_detection.FaceDetection(
            model_selection=FACE_MODEL_SELECTION,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        ) as face_detector:

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                faces = list(detect_faces(frame, face_detector))
                active_tracks, tracks, next_track_id = update_tracks(
                    faces,
                    tracks,
                    next_track_id,
                )

                for track_id, track in active_tracks:
                    x, y, w, h = int_box(track["bbox"])

                    if w <= 0 or h <= 0:
                        continue

                    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

                    if focal_length_found is not None:
                        if track["misses"] == 0:
                            distance_cm = distance_finder(
                                focal_length_found,
                                KNOWN_FACE_WIDTH_CM,
                                w,
                            )
                            smoothed = smooth_distance(
                                track["smoothed_distance"],
                                distance_cm,
                            )
                            track["smoothed_distance"] = smoothed
                        else:
                            smoothed = track["smoothed_distance"]

                        if smoothed is not None:
                            label = f"ID {track_id} | {smoothed:.1f} CM"
                        else:
                            label = f"ID {track_id}"
                    else:
                        label = f"ID {track_id}"

                    cv2.putText(
                        frame,
                        label,
                        (x, max(20, y - 10)),
                        FONT,
                        0.55,
                        GREEN,
                        2,
                    )

                cv2.line(frame, (20, 30), (460, 30), RED, 30)
                cv2.line(frame, (20, 30), (460, 30), BLACK, 26)
                if focal_length_found is None:
                    status_text = "Press c to calibrate at 60 cm | q quit"
                else:
                    status_text = "Calibrated | r reset | q quit"
                cv2.putText(frame, status_text, (30, 36), FONT, 0.55, GREEN, 2)

                cv2.imshow("Face Distance Estimator", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c") and len(faces) > 0:
                    _, _, widest_w, _ = max(faces, key=lambda f: f[2] * f[3])
                    focal_length_found = focal_length_finder(
                        CALIBRATION_DISTANCE_CM,
                        KNOWN_FACE_WIDTH_CM,
                        widest_w,
                    )
                    print(f"Calibrated focal length: {focal_length_found:.2f} px")
                if key == ord("r"):
                    focal_length_found = None
                    for track in tracks.values():
                        track["smoothed_distance"] = None
                    print("Calibration reset.")
                if key == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()