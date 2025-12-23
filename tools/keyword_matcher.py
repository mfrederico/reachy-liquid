"""Lightweight keyword-based tool dispatcher.

No LLM required - uses simple pattern matching on transcribed text.
Perfect for Raspberry Pi and other constrained devices.
"""

import re
from datetime import datetime
from typing import Callable


class KeywordToolMatcher:
    """Match keywords in text to trigger tool calls.

    Much lighter than running a second LLM for tool calling.
    Uses regex patterns to detect user intent.

    Example:
        >>> matcher = KeywordToolMatcher(camera=ptz_camera)
        >>> actions = matcher.match("can you look to the left please")
        >>> for action in actions:
        ...     print(f"Executing: {action}")
        ...     action()
    """

    def __init__(self, camera=None, detector=None):
        """Initialize matcher with camera/detector references.

        Args:
            camera: PTZCamera instance
            detector: ObjectDetector instance
        """
        self.camera = camera
        self.detector = detector

        # Define patterns and their handlers
        # Patterns are checked in order, first match wins
        # REQUIRES explicit action word: look, turn, pan, move, tilt
        self._patterns = [
            # Pan left - requires action word
            (r'\b(look|turn|pan|move)\b.*\b(left|leftward)\b', self._look_left),

            # Pan right - requires action word
            (r'\b(look|turn|pan|move)\b.*\b(right|rightward)\b', self._look_right),

            # Tilt up - requires action word
            (r'\b(look|tilt|move)\b.*\b(up|upward)\b', self._look_up),

            # Tilt down - requires action word
            (r'\b(look|tilt|move)\b.*\b(down|downward)\b', self._look_down),

            # Zoom in
            (r'\bzoom\b.*\b(in|closer)\b', self._zoom_in),
            (r'\b(closer|magnify|enlarge)\b', self._zoom_in),
            (r'\bget closer\b', self._zoom_in),

            # Zoom out
            (r'\bzoom\b.*\b(out|back|away)\b', self._zoom_out),
            (r'\b(wider|further|farther)\b.*\b(view|shot|out)\b', self._zoom_out),
            (r'\bback up\b', self._zoom_out),

            # Center/reset
            (r'\b(center|reset|home|straight ahead)\b', self._center),
            (r'\blook\b.*\b(forward|ahead|straight)\b', self._center),

            # Describe what's visible
            (r'\bwhat\b.*\b(see|seeing|visible|around)\b', self._describe),
            (r'\bdescribe\b.*\b(view|see|scene)\b', self._describe),
            (r'\b(look around|scan|survey)\b', self._describe),

            # Time queries
            (r'\bwhat\b.*\b(time)\b', self._get_time),
            (r'\b(tell|give)\b.*\b(time)\b', self._get_time),
            (r"\bwhat'?s?\b.*\b(clock)\b", self._get_time),

            # Date queries
            (r'\bwhat\b.*\b(date|day)\b', self._get_date),
            (r'\b(tell|give)\b.*\b(date|day)\b', self._get_date),
            (r"\btoday'?s?\b.*\bdate\b", self._get_date),
        ]

        # Compile patterns for efficiency
        self._compiled = [(re.compile(p, re.IGNORECASE), h) for p, h in self._patterns]

    def match(self, text: str) -> list[Callable]:
        """Find matching tool actions for the given text.

        Args:
            text: User's transcribed speech

        Returns:
            List of callable actions (may be empty)
        """
        if not text:
            return []

        actions = []
        text_lower = text.lower()

        for pattern, handler in self._compiled:
            if pattern.search(text_lower):
                actions.append(handler)
                break  # Only match first pattern

        return actions

    def process(self, text: str) -> list[dict]:
        """Match and execute any tool actions.

        Args:
            text: User's transcribed speech

        Returns:
            List of result dictionaries
        """
        results = []
        for action in self.match(text):
            result = action()
            if result:
                results.append(result)
        return results

    def has_camera(self) -> bool:
        """Check if camera is available."""
        return self.camera is not None and self.camera.has_ptz

    # ==================== Tool Handlers ====================

    def _look_left(self) -> dict:
        if not self.has_camera():
            return {"success": False, "error": "No PTZ camera"}
        print("[Camera] Looking left...")
        success = self.camera.look_left()
        return {
            "success": success,
            "action": "looked left",
            "position": self.camera.pan_position
        }

    def _look_right(self) -> dict:
        if not self.has_camera():
            return {"success": False, "error": "No PTZ camera"}
        print("[Camera] Looking right...")
        success = self.camera.look_right()
        return {
            "success": success,
            "action": "looked right",
            "position": self.camera.pan_position
        }

    def _look_up(self) -> dict:
        if not self.has_camera():
            return {"success": False, "error": "No PTZ camera"}
        print("[Camera] Looking up...")
        success = self.camera.look_up()
        return {
            "success": success,
            "action": "looked up",
            "position": self.camera.tilt_position
        }

    def _look_down(self) -> dict:
        if not self.has_camera():
            return {"success": False, "error": "No PTZ camera"}
        print("[Camera] Looking down...")
        success = self.camera.look_down()
        return {
            "success": success,
            "action": "looked down",
            "position": self.camera.tilt_position
        }

    def _zoom_in(self) -> dict:
        if not self.has_camera():
            return {"success": False, "error": "No PTZ camera"}
        print("[Camera] Zooming in...")
        success = self.camera.zoom_in()
        return {
            "success": success,
            "action": "zoomed in",
            "zoom": self.camera.zoom_level
        }

    def _zoom_out(self) -> dict:
        if not self.has_camera():
            return {"success": False, "error": "No PTZ camera"}
        print("[Camera] Zooming out...")
        success = self.camera.zoom_out()
        return {
            "success": success,
            "action": "zoomed out",
            "zoom": self.camera.zoom_level
        }

    def _center(self) -> dict:
        if not self.has_camera():
            return {"success": False, "error": "No PTZ camera"}
        print("[Camera] Centering...")
        success = self.camera.center()
        return {"success": success, "action": "centered camera"}

    def _describe(self) -> dict:
        """Describe what's currently visible."""
        if self.camera is None:
            return {"success": False, "error": "No camera"}

        frame = self.camera.get_frame()
        if frame is None:
            return {"success": False, "error": "Could not get frame"}

        if self.detector is None:
            return {
                "success": True,
                "action": "captured frame",
                "objects": [],
                "message": "No detector available"
            }

        detections = self.detector.detect(frame)
        objects = [d.get("name", "object") for d in detections]

        return {
            "success": True,
            "action": "scanned view",
            "objects": objects,
            "message": f"I can see: {', '.join(objects)}" if objects else "I don't see any recognizable objects"
        }

    def _get_time(self) -> dict:
        """Get current time with seconds."""
        now = datetime.now()
        time_str = now.strftime("%I:%M:%S %p")  # e.g., "02:30:45 PM"
        print(f"[System] Time: {time_str}")
        return {
            "success": True,
            "action": "checked time",
            "time": time_str,
            "message": f"The current time is {time_str}"
        }

    def _get_date(self) -> dict:
        """Get current date."""
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")  # e.g., "Monday, December 23, 2024"
        day_of_week = now.strftime("%A")
        print(f"[System] Date: {date_str}")
        return {
            "success": True,
            "action": "checked date",
            "date": date_str,
            "day": day_of_week,
            "message": f"Today is {date_str}"
        }


# Convenience function
def create_keyword_matcher(camera=None, detector=None) -> KeywordToolMatcher:
    """Create a keyword matcher for PTZ control.

    Args:
        camera: PTZCamera instance
        detector: ObjectDetector instance

    Returns:
        Configured KeywordToolMatcher
    """
    return KeywordToolMatcher(camera=camera, detector=detector)
