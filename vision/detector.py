"""Object detection using YOLOv8."""

import numpy as np
from ultralytics import YOLO

import config


class ObjectDetector:
    """YOLOv8 object detector with tracking support."""

    def __init__(self, model_path: str = None):
        model_path = model_path or config.YOLO_MODEL
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        print("YOLO model loaded!")

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = None,
    ) -> list[dict]:
        """
        Detect objects in a frame.

        Args:
            frame: BGR image as numpy array
            confidence: Minimum confidence threshold

        Returns:
            List of detections with 'label', 'confidence', 'box', 'position'
        """
        confidence = confidence or config.DETECTION_CONFIDENCE

        # Run inference
        results = self.model(frame, verbose=False, conf=confidence)

        detections = []
        frame_width = frame.shape[1]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().item()
                cls = int(boxes.cls[i].cpu().item())
                label = self.model.names[cls]

                # Calculate position (left, center, right)
                center_x = (box[0] + box[2]) / 2
                if center_x < frame_width / 3:
                    position = "left"
                elif center_x > 2 * frame_width / 3:
                    position = "right"
                else:
                    position = "center"

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": box.tolist(),
                    "position": position,
                })

        return detections

    def detect_and_track(
        self,
        frame: np.ndarray,
        confidence: float = None,
    ) -> list[dict]:
        """
        Detect and track objects across frames.

        Args:
            frame: BGR image as numpy array
            confidence: Minimum confidence threshold

        Returns:
            List of detections with 'label', 'confidence', 'box', 'position', 'track_id'
        """
        confidence = confidence or config.DETECTION_CONFIDENCE

        # Run tracking
        results = self.model.track(
            frame,
            verbose=False,
            conf=confidence,
            persist=True,  # Maintain track IDs across frames
        )

        detections = []
        frame_width = frame.shape[1]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().item()
                cls = int(boxes.cls[i].cpu().item())
                label = self.model.names[cls]

                # Get track ID if available
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().item())

                # Calculate position
                center_x = (box[0] + box[2]) / 2
                if center_x < frame_width / 3:
                    position = "left"
                elif center_x > 2 * frame_width / 3:
                    position = "right"
                else:
                    position = "center"

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": box.tolist(),
                    "position": position,
                    "track_id": track_id,
                })

        return detections

    def summarize_scene(self, detections: list[dict]) -> str:
        """
        Create a natural language summary of detected objects.

        Args:
            detections: List of detection dictionaries

        Returns:
            Human-readable scene description
        """
        if not detections:
            return "I don't see anything notable right now."

        # Group by label
        by_label = {}
        for det in detections:
            label = det["label"]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(det["position"])

        # Build description
        parts = []
        for label, positions in by_label.items():
            count = len(positions)
            if count == 1:
                parts.append(f"{label} ({positions[0]})")
            else:
                parts.append(f"{count} {label}s")

        return f"I can see: {', '.join(parts)}"
