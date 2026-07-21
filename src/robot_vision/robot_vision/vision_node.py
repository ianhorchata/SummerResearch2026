"""On-demand class-agnostic object detection over gscam2 camera streams.

The Jetson CSI cameras are owned by gscam2 (`nvarguscamerasrc`), which allows
only one consumer per sensor. This node subscribes to `<cam>/image_raw`, caches
the newest frame per camera, and runs detection only when `/vision/detect` is
called.

Backends (parameter ``backend``)
--------------------------------
- ``fastsam`` (default): Ultralytics FastSAM — segment anything, no class labels.
- ``blob``: classical floor-difference + contours (no model download; good for
  objects that contrast with the floor).
- ``yolo``: pretrained COCO detector (only useful if you need named classes).

All backends return the same ``vision_msgs/Detection2DArray`` so
``ground_pose_node`` keeps working unchanged. Class-agnostic hits use
``class_id = "object"``.

    ros2 launch robot_vision vision.launch.py
    ros2 launch robot_vision vision.launch.py backend:=blob
    ros2 service call /vision/detect robot_interfaces/srv/DetectObjects "{camera: 'left'}"
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

from robot_interfaces.srv import DetectObjects


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")

        self.declare_parameter("cameras", ["left", "right"])
        # fastsam | blob | yolo
        self.declare_parameter("backend", "fastsam")
        # FastSAM-s.pt / FastSAM-x.pt, or a YOLO .pt/.engine when backend=yolo
        self.declare_parameter("model", "FastSAM-s.pt")
        self.declare_parameter("confidence", 0.7)
        self.declare_parameter("device", "")
        self.declare_parameter("imgsz", 640)
        # Reject tiny noise and huge "floor/wall" segments.
        self.declare_parameter("min_area_frac", 0.001)
        self.declare_parameter("max_area_frac", 0.2)
        # Ignore detections whose center is above this fraction of image height
        # (forward cameras: objects of interest sit in the lower FOV).
        self.declare_parameter("roi_top_frac", 0.5)
        self.declare_parameter("max_detections", 15)
        # Merge near-duplicate boxes: similar area + high overlap (IoS).
        # overlap = intersection / min(area); size_ratio = max(area)/min(area).
        # Set merge_overlap_frac <= 0 to disable.
        self.declare_parameter("merge_overlap_frac", 0.80)
        self.declare_parameter("merge_size_ratio", 1.60)
        # blob backend knobs
        self.declare_parameter("blob_floor_band_frac", 0.18)
        self.declare_parameter("blob_diff_thresh", 28.0)
        self.declare_parameter("blob_blur_ksize", 5)
        self.declare_parameter("save_debug_images", True)
        self.declare_parameter("debug_image_dir", "~/ros2_ws/vision_debug")

        cameras = list(self.get_parameter("cameras").value)
        self._backend = str(self.get_parameter("backend").value).strip().lower()
        self._model_name = self.get_parameter("model").value
        self._default_conf = float(self.get_parameter("confidence").value)
        self._device = self.get_parameter("device").value or None
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._min_area_frac = float(self.get_parameter("min_area_frac").value)
        self._max_area_frac = float(self.get_parameter("max_area_frac").value)
        self._roi_top_frac = float(self.get_parameter("roi_top_frac").value)
        self._max_detections = int(self.get_parameter("max_detections").value)
        self._merge_overlap_frac = float(
            self.get_parameter("merge_overlap_frac").value)
        self._merge_size_ratio = float(
            self.get_parameter("merge_size_ratio").value)
        self._blob_floor_band = float(
            self.get_parameter("blob_floor_band_frac").value)
        self._blob_diff_thresh = float(
            self.get_parameter("blob_diff_thresh").value)
        self._blob_blur = int(self.get_parameter("blob_blur_ksize").value)
        self._save_debug = self._as_bool(
            self.get_parameter("save_debug_images").value)
        self._debug_dir = Path(
            os.path.expanduser(self.get_parameter("debug_image_dir").value)
        )

        if self._backend not in ("fastsam", "blob", "yolo"):
            self.get_logger().warn(
                f"Unknown backend '{self._backend}', falling back to 'blob'")
            self._backend = "blob"

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = {cam: None for cam in cameras}
        self._model = None
        self._model_kind: Optional[str] = None  # "fastsam" | "yolo"
        self._fastsam_predictor = None

        self._cb_group = ReentrantCallbackGroup()
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._subs = []
        for cam in cameras:
            topic = f"{cam}/image_raw"
            sub = self.create_subscription(
                Image,
                topic,
                self._make_image_cb(cam),
                sensor_qos,
                callback_group=self._cb_group,
            )
            self._subs.append(sub)
            self.get_logger().info(f"Watching camera '{cam}' on {topic}")

        self.create_service(
            DetectObjects,
            "vision/detect",
            self._on_detect,
            callback_group=self._cb_group,
        )

        if self._save_debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f"Debug images will be saved under {self._debug_dir}"
            )

        self._ensure_model()
        self.get_logger().info(
            f"vision_node ready (backend={self._backend}); "
            "call /vision/detect to run detection"
        )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _make_image_cb(self, cam):
        def _cb(msg):
            with self._lock:
                self._latest[cam] = msg
        return _cb

    def _ensure_model(self) -> Tuple[bool, str]:
        """Load FastSAM/YOLO when needed. blob needs no model."""
        if self._backend == "blob":
            return True, ""
        if self._model is not None:
            return True, ""

        try:
            from ultralytics import FastSAM, YOLO
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"Ultralytics not available ({exc}). "
                "Install: pip3 install ultralytics — or use backend:=blob"
            )
            self.get_logger().error(msg)
            if self._backend == "fastsam":
                self.get_logger().warn("Falling back to backend=blob")
                self._backend = "blob"
                return True, ""
            return False, msg

        try:
            if self._backend == "fastsam":
                name = str(self._model_name)
                # Common misconfig: robot.launch used to pass yolov8n.pt while
                # backend stayed fastsam — FastSAM(yolo.pt) returns 0 boxes.
                lower = name.lower()
                if "yolo" in lower and "fastsam" not in lower:
                    name = "FastSAM-s.pt"
                    self.get_logger().warn(
                        f"backend=fastsam but model looks like YOLO "
                        f"('{self._model_name}'); using '{name}' instead"
                    )
                self.get_logger().info(f"Loading FastSAM model '{name}' ...")
                self._model = FastSAM(name)
                # FastSAM-*.pt checkpoints often report task=detect; Ultralytics
                # then fails _smart_load because FastSAM.task_map only has
                # "segment". Force segment + patch _smart_load as a safety net.
                self._force_fastsam_segment(self._model)
                self._model_kind = "fastsam"
                self._fastsam_predictor = None
            else:
                # yolo
                name = self._model_name
                if "FastSAM" in str(name) or str(name).endswith("-s.pt"):
                    # Common misconfig when switching backends; use COCO nano.
                    name = "yolov8n.pt"
                    self.get_logger().warn(
                        f"backend=yolo but model looks like FastSAM "
                        f"('{self._model_name}'); using '{name}' instead"
                    )
                self.get_logger().info(f"Loading YOLO model '{name}' ...")
                self._model = YOLO(name)
                self._model_kind = "yolo"
            self.get_logger().info(f"{self._model_kind} model loaded")
            return True, ""
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load model '{self._model_name}': {exc}"
            self.get_logger().error(msg)
            if self._backend == "fastsam":
                self.get_logger().warn("Falling back to backend=blob")
                self._backend = "blob"
                self._model = None
                self._model_kind = None
                return True, ""
            return False, msg

    def _on_detect(self, request, response):
        cam = request.camera or "left"
        if cam not in self._latest:
            response.success = False
            response.message = (
                f"Unknown camera '{cam}'. Known cameras: "
                f"{sorted(self._latest.keys())}"
            )
            return response

        ok, err = self._ensure_model()
        if not ok:
            response.success = False
            response.message = err
            return response

        with self._lock:
            img_msg = self._latest[cam]
        if img_msg is None:
            response.success = False
            response.message = (
                f"No frame received yet on '{cam}/image_raw'. Is gscam2 running?"
            )
            return response

        try:
            frame = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"cv_bridge conversion failed: {exc}"
            return response

        conf = request.confidence if request.confidence > 0.0 else self._default_conf

        try:
            if self._backend == "blob":
                boxes, annotated = self._run_blob(frame)
                results_for_plot = None
            elif self._backend == "fastsam":
                try:
                    boxes, results_for_plot = self._run_fastsam(frame, conf)
                    annotated = None
                except Exception as sam_exc:  # noqa: BLE001
                    self.get_logger().error(
                        f"FastSAM failed ({sam_exc}); falling back to blob"
                    )
                    # Stick with blob for later calls so the service stays usable.
                    self._backend = "blob"
                    boxes, annotated = self._run_blob(frame)
                    results_for_plot = None
            else:
                boxes, results_for_plot = self._run_yolo(frame, conf)
                annotated = None
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Detection failed ({self._backend}): {exc}"
            self.get_logger().error(response.message)
            return response

        detections = self._boxes_to_detection_array(boxes, img_msg.header)
        response.detections = detections
        response.success = True
        response.message = (
            f"Detected {len(detections.detections)} object(s) on '{cam}' "
            f"via {self._backend} (conf>={conf:.2f})"
        )
        self.get_logger().info(response.message)

        if self._save_debug:
            saved = self._save_debug_images(
                cam, frame, results_for_plot, annotated, boxes)
            if saved:
                response.message = f"{response.message}; saved {saved}"

        return response

    def _run_yolo(
        self, frame: np.ndarray, conf: float
    ) -> Tuple[List[Tuple[float, float, float, float, float, str]], Any]:
        results = self._model.predict(
            frame,
            conf=conf,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )
        boxes: List[Tuple[float, float, float, float, float, str]] = []
        if not results:
            return boxes, results
        result = results[0]
        names = getattr(result, "names", None) or getattr(self._model, "names", {})
        if result.boxes is None:
            return boxes, results
        h, w = frame.shape[:2]
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) \
                else str(cls_id)
            if self._keep_box(xyxy, h, w):
                boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], score, label))
        boxes = self._limit_boxes(boxes)
        self._release_cuda()
        return boxes, None

    def _run_fastsam(
        self, frame: np.ndarray, conf: float
    ) -> Tuple[List[Tuple[float, float, float, float, float, str]], Any]:
        results = self._fastsam_infer(frame, conf)
        boxes: List[Tuple[float, float, float, float, float, str]] = []
        if not results:
            return boxes, results
        result = results[0]
        h, w = frame.shape[:2]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                score = float(box.conf[0].item()) if box.conf is not None else conf
                if self._keep_box(xyxy, h, w):
                    boxes.append(
                        (xyxy[0], xyxy[1], xyxy[2], xyxy[3], score, "object"))
        elif result.masks is not None and getattr(result.masks, "xy", None) is not None:
            for poly in result.masks.xy:
                if poly is None or len(poly) == 0:
                    continue
                xs = poly[:, 0]
                ys = poly[:, 1]
                xyxy = [float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())]
                if self._keep_box(xyxy, h, w):
                    boxes.append(
                        (xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, "object"))

        boxes = self._limit_boxes(boxes)
        # Drop GPU tensors before debug I/O — .plot() on full-res retina masks
        # was OOMing (~3GB) and we only need the filtered boxes for overlays.
        self._release_cuda()
        return boxes, None

    @staticmethod
    def _release_cuda() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _force_fastsam_segment(model: Any) -> None:
        """Keep FastSAM on task=segment even when the .pt claims detect."""
        model.task = "segment"
        try:
            model.overrides["task"] = "segment"
        except Exception:  # noqa: BLE001
            pass
        try:
            inner = getattr(model, "model", None)
            if inner is not None:
                if hasattr(inner, "task"):
                    inner.task = "segment"
                args = getattr(inner, "args", None)
                if isinstance(args, dict):
                    args["task"] = "segment"
        except Exception:  # noqa: BLE001
            pass

        # Nuclear fallback: if task is still wrong at predict-time, map any
        # missing task key back to the segment predictor.
        if not getattr(model, "_robot_fastsam_patched", False):
            original_smart_load = model._smart_load

            def _smart_load_segment(key: str):
                try:
                    return model.task_map[model.task][key]
                except Exception:  # noqa: BLE001
                    return model.task_map["segment"][key]

            model._smart_load = _smart_load_segment  # type: ignore[method-assign]
            model._robot_fastsam_patched = True
            # Keep a reference so the closure isn't the only holder (lint/debug).
            model._robot_fastsam_original_smart_load = original_smart_load

    def _fastsam_infer(self, frame: np.ndarray, conf: float) -> Any:
        """Run FastSAM everything-mode with task=segment forced.

        Prefer calling the loaded FastSAM model (with task patched). Fall back
        to FastSAMPredictor, then raise so the caller can use blob.
        """
        # retina_masks=False: full-res masks on 3280x2464 need ~3GB+ VRAM and
        # OOM the Orin (and starve the CSI cameras). We only need boxes.
        kwargs = dict(
            conf=conf,
            imgsz=self._imgsz,
            retina_masks=False,
            iou=0.9,
            verbose=False,
        )
        if self._device:
            kwargs["device"] = self._device

        # Path 1: loaded FastSAM model with forced segment task.
        if self._model is not None:
            self._force_fastsam_segment(self._model)
            try:
                return self._model.predict(frame, **kwargs)
            except Exception as model_exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"FastSAM.predict failed ({model_exc}); trying FastSAMPredictor"
                )

        # Path 2: documented FastSAMPredictor API.
        try:
            from ultralytics.models.fastsam import FastSAMPredictor

            overrides = dict(
                conf=conf,
                task="segment",
                mode="predict",
                model=str(self._model_name),
                save=False,
                imgsz=self._imgsz,
                retina_masks=False,
                iou=0.9,
                verbose=False,
            )
            if self._device:
                overrides["device"] = self._device
            if self._fastsam_predictor is None:
                self._fastsam_predictor = FastSAMPredictor(overrides=overrides)
            else:
                self._fastsam_predictor.args.conf = conf
                self._fastsam_predictor.args.imgsz = self._imgsz
                self._fastsam_predictor.args.task = "segment"
            return self._fastsam_predictor(frame)
        except Exception as pred_exc:  # noqa: BLE001
            raise RuntimeError(
                f"FastSAM inference unavailable ({pred_exc}). "
                "Try: ros2 launch robot_vision vision.launch.py backend:=blob"
            ) from pred_exc

    def _run_blob(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float, float, float, str]], np.ndarray]:
        import cv2

        h, w = frame.shape[:2]
        k = self._blob_blur
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        band_h = max(1, int(h * self._blob_floor_band))
        floor_band = blurred[h - band_h:h, :, :]
        floor_mean = floor_band.reshape(-1, 3).mean(axis=0).astype(np.float32)

        diff = np.linalg.norm(
            blurred.astype(np.float32) - floor_mean, axis=2)
        mask = (diff > self._blob_diff_thresh).astype(np.uint8) * 255

        # Ignore the floor sample band itself (often noisy at the image edge).
        mask[h - band_h:h, :] = 0
        # Ignore far-field top strip.
        top = int(h * self._roi_top_frac)
        if top > 0:
            mask[:top, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Tuple[float, float, float, float, float, str]] = []
        img_area = float(h * w)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue
            frac = area / img_area
            if frac < self._min_area_frac or frac > self._max_area_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            xyxy = [float(x), float(y), float(x + bw), float(y + bh)]
            if not self._keep_box(xyxy, h, w, apply_area=False):
                continue
            # Score: how different from floor, scaled by compactness.
            peri = cv2.arcLength(contour, True)
            compactness = (4.0 * np.pi * area / (peri * peri)) if peri > 1e-3 else 0.0
            mean_diff = float(diff[y:y + bh, x:x + bw].mean()) if bw > 0 and bh > 0 else 0.0
            score = float(np.clip(
                0.35 * compactness + 0.65 * min(1.0, mean_diff / 80.0), 0.0, 1.0))
            boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], score, "object"))

        boxes = self._limit_boxes(boxes)
        annotated = self._draw_boxes(frame.copy(), boxes, color=(0, 255, 0))
        return boxes, annotated

    def _keep_box(
        self,
        xyxy: List[float],
        h: int,
        w: int,
        apply_area: bool = True,
    ) -> bool:
        x1, y1, x2, y2 = xyxy
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 2.0 or bh < 2.0:
            return False
        cy = (y1 + y2) * 0.5
        if cy < h * self._roi_top_frac:
            return False
        if apply_area:
            frac = (bw * bh) / float(h * w)
            if frac < self._min_area_frac or frac > self._max_area_frac:
                return False
        return True

    def _limit_boxes(
        self, boxes: List[Tuple[float, float, float, float, float, str]]
    ) -> List[Tuple[float, float, float, float, float, str]]:
        boxes.sort(key=lambda b: b[4], reverse=True)
        before = len(boxes)
        boxes = self._merge_overlapping_boxes(boxes)
        if len(boxes) < before:
            self.get_logger().info(
                f"Merged overlapping boxes: {before} -> {len(boxes)}"
            )
        if self._max_detections > 0:
            boxes = boxes[: self._max_detections]
        return boxes

    def _merge_overlapping_boxes(
        self, boxes: List[Tuple[float, float, float, float, float, str]]
    ) -> List[Tuple[float, float, float, float, float, str]]:
        """Combine similar-sized, nearly-fully-overlapping boxes into one.

        Keeps the higher score and expands the box to the union of the pair.
        Boxes are assumed score-sorted (highest first).
        """
        if self._merge_overlap_frac <= 0.0 or len(boxes) < 2:
            return boxes

        kept: List[Tuple[float, float, float, float, float, str]] = []
        for box in boxes:
            merged_into = False
            for i, other in enumerate(kept):
                if self._should_merge_boxes(box, other):
                    kept[i] = self._union_boxes(other, box)
                    merged_into = True
                    break
            if not merged_into:
                kept.append(box)
        return kept

    def _should_merge_boxes(
        self,
        a: Tuple[float, float, float, float, float, str],
        b: Tuple[float, float, float, float, float, str],
    ) -> bool:
        ax1, ay1, ax2, ay2, _, _ = a
        bx1, by1, bx2, by2, _, _ = b
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if area_a <= 0.0 or area_b <= 0.0:
            return False

        size_ratio = max(area_a, area_b) / min(area_a, area_b)
        if size_ratio > self._merge_size_ratio:
            return False

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        # Intersection over smaller box ≈ "almost completely overlapped".
        ios = inter / min(area_a, area_b)
        return ios >= self._merge_overlap_frac

    @staticmethod
    def _union_boxes(
        a: Tuple[float, float, float, float, float, str],
        b: Tuple[float, float, float, float, float, str],
    ) -> Tuple[float, float, float, float, float, str]:
        """Union bbox; keep the higher score and its label."""
        ax1, ay1, ax2, ay2, ascore, alabel = a
        bx1, by1, bx2, by2, bscore, blabel = b
        if bscore > ascore:
            score, label = bscore, blabel
        else:
            score, label = ascore, alabel
        return (
            min(ax1, bx1),
            min(ay1, by1),
            max(ax2, bx2),
            max(ay2, by2),
            score,
            label,
        )

    def _boxes_to_detection_array(
        self,
        boxes: List[Tuple[float, float, float, float, float, str]],
        header,
    ) -> Detection2DArray:
        """Convert boxes to Detection2DArray with per-call numeric ids.

        ``det.id`` is ``\"0\"``, ``\"1\"``, … in score-sorted order for this
        response only (reused on the next /vision/detect call). Class name
        stays on ``hypothesis.class_id``.
        """
        arr = Detection2DArray()
        arr.header = header
        for i, (x1, y1, x2, y2, score, label) in enumerate(boxes):
            det = Detection2D()
            det.header = header

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis = ObjectHypothesis()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)

            bbox = BoundingBox2D()
            bbox.center.position.x = (x1 + x2) / 2.0
            bbox.center.position.y = (y1 + y2) / 2.0
            bbox.center.theta = 0.0
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            det.bbox = bbox
            det.id = str(i)
            arr.detections.append(det)
        return arr

    @staticmethod
    def _draw_boxes(
        image: np.ndarray,
        boxes: List[Tuple[float, float, float, float, float, str]],
        color: Tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        """Draw boxes labeled ``#id class score`` (id = enumerate order)."""
        import cv2

        out = image
        for i, (x1, y1, x2, y2, score, label) in enumerate(boxes):
            cv2.rectangle(
                out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                out,
                f"#{i} {label} {score:.2f}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return out

    def _save_debug_images(
        self,
        cam: str,
        frame: np.ndarray,
        results: Any,
        annotated: Optional[np.ndarray],
        boxes: List[Tuple[float, float, float, float, float, str]],
    ) -> str:
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Could not import cv2 to save debug images: {exc}")
            return ""

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"detect_{cam}_{stamp}"
        raw_path = self._debug_dir / f"{stem}_raw.jpg"
        ann_path = self._debug_dir / f"{stem}_annotated.jpg"
        saved: List[str] = []

        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to create debug dir: {exc}")
            return ""

        # Save raw first so we still get a frame if annotation fails.
        try:
            if cv2.imwrite(str(raw_path), frame):
                saved.append(raw_path.name)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to save raw debug image: {exc}")

        try:
            # Always draw on CPU with OpenCV. Ultralytics results[0].plot() can
            # allocate multi-GB GPU buffers for masks at full camera resolution.
            # Prefer re-drawing from boxes so numeric ids always match the
            # Detection2DArray; fall back to a pre-annotated frame if needed.
            if boxes:
                out = self._draw_boxes(frame.copy(), boxes)
            elif annotated is not None:
                out = annotated
            else:
                out = frame.copy()
            if cv2.imwrite(str(ann_path), out):
                saved.append(ann_path.name)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to save annotated debug image: {exc}")

        if not saved:
            return ""
        self.get_logger().info(f"Saved debug images: {', '.join(saved)}")
        return " + ".join(saved)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
