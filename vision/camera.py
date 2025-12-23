"""Camera capture using OpenCV."""

import cv2
import numpy as np
from threading import Thread, Lock
import time

import config


class Camera:
    """Threaded camera capture for background frame grabbing."""

    def __init__(
        self,
        camera_index: int = None,
        width: int = None,
        height: int = None,
    ):
        self.camera_index = camera_index or config.CAMERA_INDEX
        self.width = width or config.CAMERA_WIDTH
        self.height = height or config.CAMERA_HEIGHT

        self.cap = None
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.thread = None

    def start(self) -> bool:
        """
        Start camera capture in background thread.

        Returns:
            True if camera opened successfully
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_index}")
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Start capture thread
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # Wait for first frame
        time.sleep(0.5)

        print(f"Camera started ({self.width}x{self.height})")
        return True

    def _capture_loop(self):
        """Background thread that continuously captures frames."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self) -> np.ndarray | None:
        """
        Get the most recent frame.

        Returns:
            BGR frame as numpy array, or None if no frame available
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("Camera stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
