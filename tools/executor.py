"""Tool executor for parsing and executing LFM2-Tool function calls."""

import re
import json
import ast
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


class ToolExecutor:
    """Executes tool calls from LFM2-Tool model output.

    Parses function calls in the format:
        <|tool_call_start|>[function_name(param="value")]<|tool_call_end|>

    And returns responses in the format:
        <|tool_response_start|>{result_json}<|tool_response_end|>

    Example:
        >>> executor = ToolExecutor(camera=ptz_camera, detector=yolo_detector)
        >>> response = executor.execute("look_left", {})
        >>> print(response)
        {"success": true, "message": "Panned camera left"}
    """

    # Pattern to extract tool calls from model output
    TOOL_CALL_PATTERN = re.compile(
        r'<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>',
        re.DOTALL
    )

    def __init__(self, camera=None, detector=None):
        """Initialize the tool executor.

        Args:
            camera: PTZCamera instance for camera controls
            detector: ObjectDetector instance for vision
        """
        self.camera = camera
        self.detector = detector

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains a tool call."""
        return '<|tool_call_start|>' in text

    def extract_tool_calls(self, text: str) -> list[tuple[str, dict]]:
        """Extract all tool calls from model output.

        Args:
            text: Model output text

        Returns:
            List of (function_name, arguments) tuples
        """
        calls = []
        matches = self.TOOL_CALL_PATTERN.findall(text)

        for match in matches:
            try:
                # Parse Pythonic function call syntax
                # e.g., "look_left(amount=10)" or "look_center()"
                call_str = match.strip()

                # Handle multiple calls separated by comma
                for single_call in self._split_calls(call_str):
                    func_name, args = self._parse_call(single_call)
                    if func_name:
                        calls.append((func_name, args))
            except Exception as e:
                print(f"Failed to parse tool call '{match}': {e}")

        return calls

    def _split_calls(self, call_str: str) -> list[str]:
        """Split multiple function calls."""
        # Simple split - assumes no nested parentheses
        calls = []
        depth = 0
        current = ""

        for char in call_str:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                if current.strip():
                    calls.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            calls.append(current.strip())

        return calls

    def _parse_call(self, call_str: str) -> tuple[str, dict]:
        """Parse a single function call string.

        Args:
            call_str: e.g., "look_left(amount=10)" or "look_center()"

        Returns:
            (function_name, arguments_dict)
        """
        # Match function_name(args)
        match = re.match(r'(\w+)\s*\((.*)\)', call_str, re.DOTALL)
        if not match:
            return None, {}

        func_name = match.group(1)
        args_str = match.group(2).strip()

        if not args_str:
            return func_name, {}

        # Parse keyword arguments
        args = {}
        try:
            # Try to parse as Python literal
            # Wrap in dict() call to parse keyword arguments
            eval_str = f"dict({args_str})"
            args = eval(eval_str)
        except Exception:
            # Fallback: manual parsing
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    # Try to convert to int
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                    args[key] = value

        return func_name, args

    def execute(self, func_name: str, args: dict) -> dict:
        """Execute a tool function.

        Args:
            func_name: Name of the function to call
            args: Arguments dictionary

        Returns:
            Result dictionary
        """
        # Map function names to implementations
        handlers = {
            "look_left": self._look_left,
            "look_right": self._look_right,
            "look_up": self._look_up,
            "look_down": self._look_down,
            "zoom_in": self._zoom_in,
            "zoom_out": self._zoom_out,
            "look_center": self._look_center,
            "get_camera_status": self._get_camera_status,
            "describe_view": self._describe_view,
            "get_time": self._get_time,
            "get_date": self._get_date,
            "get_time_in_location": self._get_time_in_location,
            "get_time_zone": self._get_time_in_location,  # Alias for model flexibility
        }

        handler = handlers.get(func_name)
        if handler is None:
            return {"success": False, "error": f"Unknown function: {func_name}"}

        try:
            return handler(**args)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_and_format(self, func_name: str, args: dict) -> str:
        """Execute a tool and return formatted response for LFM2-Tool.

        Args:
            func_name: Name of the function to call
            args: Arguments dictionary

        Returns:
            Formatted response string with special tokens
        """
        result = self.execute(func_name, args)
        return f"<|tool_response_start|>{json.dumps(result)}<|tool_response_end|>"

    def process_output(self, text: str) -> tuple[str, list[str]]:
        """Process model output, executing any tool calls.

        Args:
            text: Model output text

        Returns:
            (clean_text, list_of_tool_responses)
        """
        if not self.has_tool_call(text):
            return text, []

        responses = []
        calls = self.extract_tool_calls(text)

        for func_name, args in calls:
            print(f"[Tool] Executing {func_name}({args})")
            response = self.execute_and_format(func_name, args)
            responses.append(response)

        # Remove tool call tokens from text
        clean_text = self.TOOL_CALL_PATTERN.sub('', text).strip()

        return clean_text, responses

    # ==================== Tool Implementations ====================

    def _check_camera(self) -> dict | None:
        """Check if camera is available."""
        if self.camera is None:
            return {"success": False, "error": "Camera not available"}
        if not self.camera.has_ptz:
            return {"success": False, "error": "Camera does not have PTZ controls"}
        return None

    def _look_left(self, amount: int = None) -> dict:
        error = self._check_camera()
        if error:
            return error

        success = self.camera.look_left(amount)
        if success:
            return {"success": True, "action": "panned left", "new_pan": self.camera.pan_position}
        return {"success": False, "error": "Failed to pan left"}

    def _look_right(self, amount: int = None) -> dict:
        error = self._check_camera()
        if error:
            return error

        success = self.camera.look_right(amount)
        if success:
            return {"success": True, "action": "panned right", "new_pan": self.camera.pan_position}
        return {"success": False, "error": "Failed to pan right"}

    def _look_up(self, amount: int = None) -> dict:
        error = self._check_camera()
        if error:
            return error

        success = self.camera.look_up(amount)
        if success:
            return {"success": True, "action": "tilted up", "new_tilt": self.camera.tilt_position}
        return {"success": False, "error": "Failed to tilt up"}

    def _look_down(self, amount: int = None) -> dict:
        error = self._check_camera()
        if error:
            return error

        success = self.camera.look_down(amount)
        if success:
            return {"success": True, "action": "tilted down", "new_tilt": self.camera.tilt_position}
        return {"success": False, "error": "Failed to tilt down"}

    def _zoom_in(self, amount: int = None) -> dict:
        error = self._check_camera()
        if error:
            return error

        success = self.camera.zoom_in(amount)
        if success:
            return {"success": True, "action": "zoomed in", "new_zoom": self.camera.zoom_level}
        return {"success": False, "error": "Failed to zoom in"}

    def _zoom_out(self, amount: int = None) -> dict:
        error = self._check_camera()
        if error:
            return error

        success = self.camera.zoom_out(amount)
        if success:
            return {"success": True, "action": "zoomed out", "new_zoom": self.camera.zoom_level}
        return {"success": False, "error": "Failed to zoom out"}

    def _look_center(self) -> dict:
        error = self._check_camera()
        if error:
            return error

        success = self.camera.center()
        if success:
            return {"success": True, "action": "centered camera"}
        return {"success": False, "error": "Failed to center camera"}

    def _get_camera_status(self) -> dict:
        if self.camera is None:
            return {"success": False, "error": "Camera not available"}

        status = self.camera.get_ptz_status()
        return {"success": True, **status}

    def _describe_view(self) -> dict:
        """Take a snapshot and describe what's visible."""
        if self.camera is None:
            return {"success": False, "error": "Camera not available"}

        frame = self.camera.get_frame()
        if frame is None:
            return {"success": False, "error": "Could not capture frame"}

        if self.detector is None:
            return {
                "success": True,
                "message": "Frame captured but no object detector available",
                "objects": []
            }

        # Detect objects
        detections = self.detector.detect(frame)

        if not detections:
            return {
                "success": True,
                "message": "No objects detected in current view",
                "objects": []
            }

        # Format detections
        objects = []
        for det in detections:
            obj = {
                "name": det.get("name", "unknown"),
                "confidence": round(det.get("confidence", 0), 2),
                "position": det.get("position", "center")
            }
            objects.append(obj)

        return {
            "success": True,
            "message": f"Detected {len(objects)} object(s)",
            "objects": objects
        }

    def _get_time(self) -> dict:
        """Get current time."""
        now = datetime.now()
        time_str = now.strftime("%I:%M:%S %p")  # e.g., "11:54:23 AM"
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
        return {
            "success": True,
            "action": "checked date",
            "date": date_str,
            "day": now.strftime("%A"),
            "message": f"Today is {date_str}"
        }

    # Common location to timezone mapping
    LOCATION_TIMEZONES = {
        # US timezones
        "eastern": "America/New_York",
        "est": "America/New_York",
        "edt": "America/New_York",
        "central": "America/Chicago",
        "cst": "America/Chicago",
        "cdt": "America/Chicago",
        "mountain": "America/Denver",
        "mst": "America/Denver",
        "mdt": "America/Denver",
        "pacific": "America/Los_Angeles",
        "pst": "America/Los_Angeles",
        "pdt": "America/Los_Angeles",
        "alaska": "America/Anchorage",
        "hawaii": "Pacific/Honolulu",
        # US cities
        "new york": "America/New_York",
        "nyc": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "la": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "denver": "America/Denver",
        "phoenix": "America/Phoenix",
        "seattle": "America/Los_Angeles",
        "miami": "America/New_York",
        "boston": "America/New_York",
        "san francisco": "America/Los_Angeles",
        # International
        "london": "Europe/London",
        "uk": "Europe/London",
        "paris": "Europe/Paris",
        "berlin": "Europe/Berlin",
        "tokyo": "Asia/Tokyo",
        "japan": "Asia/Tokyo",
        "beijing": "Asia/Shanghai",
        "shanghai": "Asia/Shanghai",
        "china": "Asia/Shanghai",
        "sydney": "Australia/Sydney",
        "melbourne": "Australia/Melbourne",
        "australia": "Australia/Sydney",
        "dubai": "Asia/Dubai",
        "singapore": "Asia/Singapore",
        "hong kong": "Asia/Hong_Kong",
        "seoul": "Asia/Seoul",
        "korea": "Asia/Seoul",
        "mumbai": "Asia/Kolkata",
        "india": "Asia/Kolkata",
        "moscow": "Europe/Moscow",
        "russia": "Europe/Moscow",
        "bali": "Asia/Makassar",
        "indonesia": "Asia/Jakarta",
        "jakarta": "Asia/Jakarta",
        "bangkok": "Asia/Bangkok",
        "thailand": "Asia/Bangkok",
        "amsterdam": "Europe/Amsterdam",
        "rome": "Europe/Rome",
        "madrid": "Europe/Madrid",
        "lisbon": "Europe/Lisbon",
        "cairo": "Africa/Cairo",
        "johannesburg": "Africa/Johannesburg",
        "toronto": "America/Toronto",
        "vancouver": "America/Vancouver",
        "mexico city": "America/Mexico_City",
        "mexico": "America/Mexico_City",
        "sao paulo": "America/Sao_Paulo",
        "brazil": "America/Sao_Paulo",
        "buenos aires": "America/Argentina/Buenos_Aires",
        "argentina": "America/Argentina/Buenos_Aires",
        # Generic
        "utc": "UTC",
        "gmt": "UTC",
    }

    def _get_time_in_location(self, location: str = None, time_zone: str = None) -> dict:
        """Get time in a specific location or timezone."""
        # Handle both 'location' and 'time_zone' parameter names
        loc = (location or time_zone or "").lower().strip()

        if not loc:
            return {"success": False, "error": "No location specified"}

        # Try to find timezone
        tz_name = self.LOCATION_TIMEZONES.get(loc)

        # If not in mapping, try as direct timezone name
        if not tz_name:
            # Try common formats
            possible_names = [
                loc,
                loc.replace(" ", "_"),
                f"America/{loc.title().replace(' ', '_')}",
                f"Europe/{loc.title().replace(' ', '_')}",
                f"Asia/{loc.title().replace(' ', '_')}",
            ]
            for name in possible_names:
                try:
                    ZoneInfo(name)
                    tz_name = name
                    break
                except Exception:
                    continue

        if not tz_name:
            return {
                "success": False,
                "error": f"Unknown location: {location or time_zone}. Try a city name like 'Tokyo' or timezone like 'PST'."
            }

        try:
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)
            time_str = now.strftime("%I:%M:%S %p")
            date_str = now.strftime("%A, %B %d")

            return {
                "success": True,
                "action": f"checked time in {location or time_zone}",
                "time": time_str,
                "date": date_str,
                "timezone": tz_name,
                "message": f"The time in {location or time_zone} is {time_str} on {date_str}"
            }
        except Exception as e:
            return {"success": False, "error": f"Timezone error: {e}"}
