"""
Realtime people counting with a custom fire-smoke-human YOLO model.

Goal
----
This script adopts the standard "people detection and counting" logic:
    webcam frame -> YOLO inference -> filter human boxes -> count humans per frame

but it keeps the user's custom model classes:
    fire, human, smoke

It also adds fire-aware counting:
    - each human has its own bounding box
    - each fire has its own bounding box
    - each smoke has its own bounding box
    - each fire box reports how many humans overlap with it
    - the global HUD reports how many unique humans are inside/affected by fire

Expected model:
    firesmokehuman.pt

Install:
    pip install ultralytics opencv-python numpy

Run:
    python webcam_custom_model_people_counting.py --model firesmokehuman.pt --camera 0

Recommended first test:
    python webcam_custom_model_people_counting.py \
        --model firesmokehuman.pt \
        --camera 0 \
        --conf-human 0.35 \
        --conf-fire 0.30 \
        --conf-smoke 0.25 \
        --nms-iou 0.55

Keys:
    q / ESC : quit
    s       : save current annotated frame
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Ultralytics is not installed. Install it first with:\n"
        "    pip install ultralytics opencv-python numpy\n"
    ) from exc


BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: BBox
    display_id: int = -1


CLASS_ALIASES = {
    "fire": {"fire", "flame", "flames"},
    "human": {"human", "person", "people", "man", "woman"},
    "smoke": {"smoke", "smog"},
}

COLORS = {
    "fire": (0, 0, 255),          # red, BGR
    "human": (0, 255, 0),        # green, BGR
    "smoke": (160, 160, 160),    # gray, BGR
    "human_fire": (0, 255, 255), # yellow, BGR
    "panel_bg": (25, 25, 25),
    "white": (255, 255, 255),
}


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def canonical_class(name: str) -> str | None:
    normalized = normalize_name(name)
    for canonical, aliases in CLASS_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def bbox_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def center_in_box(inner: BBox, outer: BBox) -> bool:
    x1, y1, x2, y2 = inner
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    ox1, oy1, ox2, oy2 = outer
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def human_fire_overlap_ratio(human_box: BBox, fire_box: BBox) -> float:
    """Return intersection area normalized by human-box area.

    This is intentionally not standard IoU.

    For the question "how many humans are inside/affected by fire?", the useful
    quantity is how much of the human box is covered by the fire box:

        overlap_ratio = area(human box ∩ fire box) / area(human box)

    If the fire box is large and surrounds the human, IoU may be small, but this
    ratio will be high. That makes it more suitable for emergency/fire logic.
    """
    human_area = bbox_area(human_box)
    if human_area <= 0.0:
        return 0.0
    return intersection_area(human_box, fire_box) / human_area


def is_human_in_fire(
    human_box: BBox,
    fire_box: BBox,
    min_overlap_human: float,
    use_center_test: bool = True,
) -> bool:
    """Decide whether one human detection is inside/affected by one fire box."""
    ratio = human_fire_overlap_ratio(human_box, fire_box)
    if ratio >= min_overlap_human:
        return True
    if use_center_test and center_in_box(human_box, fire_box):
        return True
    return False


def parse_yolo_detections(result, model_names: Dict[int, str]) -> List[Detection]:
    detections: List[Detection] = []
    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, cls_ids):
        raw_name = model_names.get(int(cls_id), str(cls_id))
        cls_name = canonical_class(raw_name)

        if cls_name is None:
            continue

        detections.append(
            Detection(
                cls_id=int(cls_id),
                cls_name=cls_name,
                conf=float(conf),
                box=tuple(float(v) for v in box),
            )
        )

    return detections


def apply_class_specific_confidence(
    detections: List[Detection],
    conf_human: float,
    conf_fire: float,
    conf_smoke: float,
) -> List[Detection]:
    thresholds = {
        "human": conf_human,
        "fire": conf_fire,
        "smoke": conf_smoke,
    }

    return [
        det for det in detections
        if det.conf >= thresholds.get(det.cls_name, 1.0)
    ]


def split_and_number_detections(
    detections: List[Detection],
) -> Tuple[List[Detection], List[Detection], List[Detection]]:
    """Split detections into humans, fires, smokes and assign display IDs.

    This is the core people-counting logic:
        humans = all detections whose class is "human"
        num_people = len(humans)

    The display IDs H1, H2, ... are frame-local IDs, not tracking IDs.
    """
    humans = [det for det in detections if det.cls_name == "human"]
    fires = [det for det in detections if det.cls_name == "fire"]
    smokes = [det for det in detections if det.cls_name == "smoke"]

    for i, det in enumerate(humans, start=1):
        det.display_id = i
    for i, det in enumerate(fires, start=1):
        det.display_id = i
    for i, det in enumerate(smokes, start=1):
        det.display_id = i

    return humans, fires, smokes


def assign_humans_to_fire_boxes(
    humans: List[Detection],
    fires: List[Detection],
    min_overlap_human: float,
    use_center_test: bool = True,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]], List[int]]:
    """Assign frame-local human IDs to fire boxes.

    Returns
    -------
    fire_to_human_ids:
        {fire_display_id: [human_display_id, ...]}

    fire_to_overlap_ratio:
        {fire_display_id: {human_display_id: overlap_ratio, ...}}

    human_ids_in_any_fire:
        sorted unique human IDs affected by at least one fire box.
    """
    fire_to_human_ids: Dict[int, List[int]] = {}
    fire_to_overlap_ratio: Dict[int, Dict[int, float]] = {}
    human_ids_in_any_fire = set()

    for fire in fires:
        fire_to_human_ids[fire.display_id] = []
        fire_to_overlap_ratio[fire.display_id] = {}

        for human in humans:
            ratio = human_fire_overlap_ratio(human.box, fire.box)
            inside = is_human_in_fire(
                human_box=human.box,
                fire_box=fire.box,
                min_overlap_human=min_overlap_human,
                use_center_test=use_center_test,
            )

            if inside:
                fire_to_human_ids[fire.display_id].append(human.display_id)
                fire_to_overlap_ratio[fire.display_id][human.display_id] = ratio
                human_ids_in_any_fire.add(human.display_id)

    return fire_to_human_ids, fire_to_overlap_ratio, sorted(human_ids_in_any_fire)


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y - th - baseline - 5)

    cv2.rectangle(
        frame,
        (x, y_top),
        (x + tw + 8, y_top + th + baseline + 7),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y_top + th + 2),
        font,
        scale,
        COLORS["white"],
        thickness,
        cv2.LINE_AA,
    )


def draw_box(
    frame: np.ndarray,
    box: BBox,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_label(frame, label, x1, y1, color)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    num_people: int,
    num_fire: int,
    num_smoke: int,
    humans_in_fire_ids: List[int],
) -> None:
    panel_lines = [
        f"FPS: {fps:.1f}",
        f"People Count: {num_people}",
        f"Fire Boxes: {num_fire}",
        f"Smoke Boxes: {num_smoke}",
        f"People in Fire: {len(humans_in_fire_ids)}",
    ]

    if humans_in_fire_ids:
        ids = ", ".join(f"H{i}" for i in humans_in_fire_ids)
        panel_lines.append(f"Affected IDs: {ids}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2

    x0, y0 = 12, 16
    line_height = 28
    max_width = 0

    for line in panel_lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, tw)

    cv2.rectangle(
        frame,
        (x0 - 6, y0 - 4),
        (x0 + max_width + 16, y0 + line_height * len(panel_lines) + 4),
        COLORS["panel_bg"],
        -1,
    )

    y = y0 + 22
    for line in panel_lines:
        cv2.putText(
            frame,
            line,
            (x0, y),
            font,
            scale,
            COLORS["white"],
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def draw_all_detections(
    frame: np.ndarray,
    humans: List[Detection],
    fires: List[Detection],
    smokes: List[Detection],
    fire_to_human_ids: Dict[int, List[int]],
    humans_in_fire_ids: List[int],
) -> None:
    humans_in_fire_set = set(humans_in_fire_ids)

    # Draw fire first, then smoke, then humans.
    # Human boxes are drawn last so that they remain readable when overlapping fire.
    for fire in fires:
        human_ids = fire_to_human_ids.get(fire.display_id, [])
        label = f"fire F{fire.display_id} {fire.conf:.2f} | humans: {len(human_ids)}"
        draw_box(frame, fire.box, label, COLORS["fire"], thickness=2)

    for smoke in smokes:
        label = f"smoke S{smoke.display_id} {smoke.conf:.2f}"
        draw_box(frame, smoke.box, label, COLORS["smoke"], thickness=2)

    for human in humans:
        in_fire = human.display_id in humans_in_fire_set
        color = COLORS["human_fire"] if in_fire else COLORS["human"]
        suffix = " IN_FIRE" if in_fire else ""
        label = f"human H{human.display_id} {human.conf:.2f}{suffix}"
        draw_box(frame, human.box, label, color, thickness=3 if in_fire else 2)


def save_jsonl(log_path: Path, record: dict) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_summary_json(summary_path: Path, record: dict) -> None:
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


class FireDetectionNode(Node):

    def __init__(self):
        super().__init__("fire_detection_node")

        self.bridge = CvBridge()

        model_path = "firesmokehuman.pt"

        self.model = YOLO(model_path)

        self.get_logger().info(f"Loaded model: {model_path}")
        self.get_logger().info(f"Classes: {self.model.names}")

        self.image_sub = self.create_subscription(
            Image,
            "/anafi/camera/image",
            self.image_callback,
            10,
        )

        self.image_pub = self.create_publisher(
            Image,
            "/anafi/fire_detection/image",
            10,
        )

        self.previous_time = time.perf_counter()
        self.fps = 0.0

        # Detection parameters
        self.conf = 0.05
        self.conf_human = 0.35
        self.conf_fire = 0.30
        self.conf_smoke = 0.25
        self.nms_iou = 0.50
        self.imgsz = 416
        self.max_det = 50
        self.min_overlap_human = 0.10

    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(str(e))
            return

        inference_conf = min(
            self.conf,
            self.conf_human,
            self.conf_fire,
            self.conf_smoke,
        )

        result = self.model.predict(
            source=frame,
            conf=inference_conf,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            verbose=False,
            max_det=self.max_det,
        )[0]

        detections = parse_yolo_detections(
            result,
            self.model.names,
        )

        detections = apply_class_specific_confidence(
            detections,
            self.conf_human,
            self.conf_fire,
            self.conf_smoke,
        )

        humans, fires, smokes = split_and_number_detections(
            detections
        )

        (
            fire_to_human_ids,
            fire_to_overlap_ratio,
            humans_in_fire_ids,
        ) = assign_humans_to_fire_boxes(
            humans,
            fires,
            self.min_overlap_human,
            True,
        )

        now = time.perf_counter()
        dt = now - self.previous_time
        self.previous_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = (
                instant_fps
                if self.fps <= 0
                else 0.9 * self.fps + 0.1 * instant_fps
            )

        draw_all_detections(
            frame,
            humans,
            fires,
            smokes,
            fire_to_human_ids,
            humans_in_fire_ids,
        )

        draw_hud(
            frame,
            self.fps,
            len(humans),
            len(fires),
            len(smokes),
            humans_in_fire_ids,
        )

        out_msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8",
        )
        out_msg.header = msg.header
        self.image_pub.publish(out_msg)

def main(args=None):

    rclpy.init(args=args)
    node = FireDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
