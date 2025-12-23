"""Camera capture with optional PTZ control.

Supports both basic OpenCV cameras and NexiGo PTZ cameras.
When a NexiGo camera is detected, PTZ controls are available.
"""

import sys
import cv2
import numpy as np
from threading import Thread, Lock
import time
from pathlib import Path

import config

# Add nexigo package to path
_nexigo_path = Path(__file__).parent.parent / "nexigo" / "src"
if str(_nexigo_path) not in sys.path:
    sys.path.insert(0, str(_nexigo_path))


class Camera:
    """Threaded camera capture for background frame grabbing.

    This is the base camera class using standard OpenCV.
    For PTZ support, use PTZCamera instead.
    """

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

    @property
    def has_ptz(self) -> bool:
        """Check if camera has PTZ capabilities."""
        return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PTZCamera(Camera):
    """Camera with Pan-Tilt-Zoom control for NexiGo PTZ cameras.

    Extends the base Camera class with PTZ control capabilities.
    Uses the NexiGo camera driver for V4L2-based PTZ control while
    maintaining the same frame capture interface.

    If a NexiGo camera is not found, falls back to basic camera mode.

    Example:
        >>> camera = PTZCamera()
        >>> if camera.start():
        ...     if camera.has_ptz:
        ...         camera.look_left()
        ...         camera.zoom_in()
        ...     frame = camera.get_frame()
    """

    def __init__(
        self,
        camera_index: int = None,
        width: int = None,
        height: int = None,
        auto_detect_nexigo: bool = True,
    ):
        super().__init__(camera_index, width, height)
        self._nexigo = None
        self._ptz_available = False
        self._auto_detect = auto_detect_nexigo

        # PTZ limits (populated on start)
        self._pan_range = None
        self._tilt_range = None
        self._zoom_range = None

        # PTZ movement amounts - will be set based on camera limits
        # Default to ~1/3 of full range for a meaningful "look left/right"
        self.pan_step = 36000   # Will be adjusted based on camera
        self.tilt_step = 36000  # Will be adjusted based on camera
        self.zoom_step = 50     # Will be adjusted based on camera

    def start(self) -> bool:
        """Start camera with PTZ detection."""
        # Try to use NexiGo camera if auto-detect enabled
        if self._auto_detect:
            try:
                from nexigo_camera import NexigoCamera

                # Try to find and open NexiGo camera
                devices = NexigoCamera.find_devices()
                if devices:
                    device = devices[0]
                    print(f"Found NexiGo PTZ camera: {device}")

                    # Use NexiGo camera
                    self._nexigo = NexigoCamera(
                        device=device,
                        resolution=type('R', (), {'width': self.width, 'height': self.height})(),
                        fps=30,
                    )
                    self._nexigo.open()
                    self._ptz_available = True

                    # Start threaded capture using NexiGo's internal OpenCV capture
                    self.cap = self._nexigo._capture
                    self.running = True
                    self.thread = Thread(target=self._capture_loop, daemon=True)
                    self.thread.start()

                    # Wait for first frame
                    time.sleep(0.5)

                    # Get PTZ limits and calculate step sizes
                    try:
                        limits = self._nexigo.get_ptz_limits()

                        # Calculate step sizes as ~1/3 of full range (meaningful movement)
                        if limits.get('pan'):
                            pan_info = limits['pan']
                            self._pan_range = (pan_info.minimum, pan_info.maximum)
                            pan_total = pan_info.maximum - pan_info.minimum
                            self.pan_step = pan_total // 3  # 1/3 of range
                            print(f"Pan range: {pan_info.minimum} to {pan_info.maximum}, step: {self.pan_step}")

                        if limits.get('tilt'):
                            tilt_info = limits['tilt']
                            self._tilt_range = (tilt_info.minimum, tilt_info.maximum)
                            tilt_total = tilt_info.maximum - tilt_info.minimum
                            self.tilt_step = tilt_total // 3  # 1/3 of range
                            print(f"Tilt range: {tilt_info.minimum} to {tilt_info.maximum}, step: {self.tilt_step}")

                        if limits.get('zoom'):
                            zoom_info = limits['zoom']
                            self._zoom_range = (zoom_info.minimum, zoom_info.maximum)
                            zoom_total = zoom_info.maximum - zoom_info.minimum
                            self.zoom_step = max(zoom_total // 5, 10)  # 1/5 of range, min 10
                            print(f"Zoom range: {zoom_info.minimum} to {zoom_info.maximum}, step: {self.zoom_step}")

                    except Exception as e:
                        print(f"Could not query PTZ limits: {e}")

                    print(f"PTZ Camera started ({self.width}x{self.height})")
                    return True

            except ImportError as e:
                print(f"NexiGo driver not available: {e}")
            except Exception as e:
                print(f"Failed to initialize NexiGo camera: {e}")
                if self._nexigo:
                    try:
                        self._nexigo.close()
                    except Exception:
                        pass
                    self._nexigo = None

        # Fall back to basic camera
        print("Using basic camera (no PTZ)")
        return super().start()

    def stop(self):
        """Stop camera and release PTZ resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if self._nexigo:
            try:
                self._nexigo.close()
            except Exception:
                pass
            self._nexigo = None
            self.cap = None  # Already closed by NexiGo
        elif self.cap:
            self.cap.release()

        self._ptz_available = False
        print("PTZ Camera stopped.")

    @property
    def has_ptz(self) -> bool:
        """Check if PTZ controls are available."""
        return self._ptz_available and self._nexigo is not None

    # ==================== PTZ Control Methods ====================

    def pan_left(self, amount: int = None) -> bool:
        """Pan camera left."""
        if not self.has_ptz:
            return False
        try:
            # Try relative first, fall back to absolute
            try:
                self._nexigo.pan_relative(-(amount or self.pan_step))
            except Exception:
                current = self._nexigo.pan
                self._nexigo.pan = current - (amount or self.pan_step)
            return True
        except Exception as e:
            print(f"Pan left failed: {e}")
            return False

    def pan_right(self, amount: int = None) -> bool:
        """Pan camera right."""
        if not self.has_ptz:
            return False
        try:
            # Try relative first, fall back to absolute
            try:
                self._nexigo.pan_relative(amount or self.pan_step)
            except Exception:
                current = self._nexigo.pan
                self._nexigo.pan = current + (amount or self.pan_step)
            return True
        except Exception as e:
            print(f"Pan right failed: {e}")
            return False

    def tilt_up(self, amount: int = None) -> bool:
        """Tilt camera up."""
        if not self.has_ptz:
            return False
        try:
            # Try relative first, fall back to absolute
            try:
                self._nexigo.tilt_relative(amount or self.tilt_step)
            except Exception:
                # Use absolute positioning
                current = self._nexigo.tilt
                self._nexigo.tilt = current + (amount or self.tilt_step)
            return True
        except Exception as e:
            print(f"Tilt up failed: {e}")
            return False

    def tilt_down(self, amount: int = None) -> bool:
        """Tilt camera down."""
        if not self.has_ptz:
            return False
        try:
            # Try relative first, fall back to absolute
            try:
                self._nexigo.tilt_relative(-(amount or self.tilt_step))
            except Exception:
                # Use absolute positioning
                current = self._nexigo.tilt
                self._nexigo.tilt = current - (amount or self.tilt_step)
            return True
        except Exception as e:
            print(f"Tilt down failed: {e}")
            return False

    def zoom_in(self, amount: int = None) -> bool:
        """Zoom camera in."""
        if not self.has_ptz:
            return False
        try:
            # Try relative first, fall back to absolute
            try:
                self._nexigo.zoom_relative(amount or self.zoom_step)
            except Exception:
                current = self._nexigo.zoom
                self._nexigo.zoom = current + (amount or self.zoom_step)
            return True
        except Exception as e:
            print(f"Zoom in failed: {e}")
            return False

    def zoom_out(self, amount: int = None) -> bool:
        """Zoom camera out."""
        if not self.has_ptz:
            return False
        try:
            # Try relative first, fall back to absolute
            try:
                self._nexigo.zoom_relative(-(amount or self.zoom_step))
            except Exception:
                current = self._nexigo.zoom
                self._nexigo.zoom = current - (amount or self.zoom_step)
            return True
        except Exception as e:
            print(f"Zoom out failed: {e}")
            return False

    def center(self) -> bool:
        """Reset camera to center/home position."""
        if not self.has_ptz:
            return False
        try:
            self._nexigo.home()
            return True
        except Exception as e:
            print(f"Center failed: {e}")
            return False

    # Aliases for more natural language
    def look_left(self, amount: int = None) -> bool:
        """Alias for pan_left."""
        return self.pan_left(amount)

    def look_right(self, amount: int = None) -> bool:
        """Alias for pan_right."""
        return self.pan_right(amount)

    def look_up(self, amount: int = None) -> bool:
        """Alias for tilt_up."""
        return self.tilt_up(amount)

    def look_down(self, amount: int = None) -> bool:
        """Alias for tilt_down."""
        return self.tilt_down(amount)

    def look_center(self) -> bool:
        """Alias for center."""
        return self.center()

    # ==================== Position Getters ====================

    @property
    def pan_position(self) -> int | None:
        """Get current pan position."""
        if not self.has_ptz:
            return None
        try:
            return self._nexigo.pan
        except Exception:
            return None

    @property
    def tilt_position(self) -> int | None:
        """Get current tilt position."""
        if not self.has_ptz:
            return None
        try:
            return self._nexigo.tilt
        except Exception:
            return None

    @property
    def zoom_level(self) -> int | None:
        """Get current zoom level."""
        if not self.has_ptz:
            return None
        try:
            return self._nexigo.zoom
        except Exception:
            return None

    def get_ptz_status(self) -> dict:
        """Get current PTZ status as a dictionary."""
        return {
            "has_ptz": self.has_ptz,
            "pan": self.pan_position,
            "tilt": self.tilt_position,
            "zoom": self.zoom_level,
        }
