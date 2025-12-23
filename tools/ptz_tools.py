"""PTZ camera tool definitions for LFM2-Tool.

These tools allow the LLM to control the PTZ camera through function calls.
"""

import json

# Tool definitions in LFM2-Tool JSON format
PTZ_TOOLS = [
    {
        "name": "look_left",
        "description": "Pan the camera to the left to see what's on the left side",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "How far to pan (optional, default is a moderate step)"
                }
            },
            "required": []
        }
    },
    {
        "name": "look_right",
        "description": "Pan the camera to the right to see what's on the right side",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "How far to pan (optional, default is a moderate step)"
                }
            },
            "required": []
        }
    },
    {
        "name": "look_up",
        "description": "Tilt the camera up to see what's above",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "How far to tilt (optional, default is a moderate step)"
                }
            },
            "required": []
        }
    },
    {
        "name": "look_down",
        "description": "Tilt the camera down to see what's below",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "How far to tilt (optional, default is a moderate step)"
                }
            },
            "required": []
        }
    },
    {
        "name": "zoom_in",
        "description": "Zoom in to see something closer or in more detail",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "How much to zoom in (optional, default is a moderate step)"
                }
            },
            "required": []
        }
    },
    {
        "name": "zoom_out",
        "description": "Zoom out to see a wider view of the scene",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "How much to zoom out (optional, default is a moderate step)"
                }
            },
            "required": []
        }
    },
    {
        "name": "look_center",
        "description": "Reset the camera to center/home position, looking straight ahead",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_camera_status",
        "description": "Get the current camera position and status (pan, tilt, zoom levels)",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "describe_view",
        "description": "Take a snapshot and describe what objects are currently visible in the camera view",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_time",
        "description": "Get the current time. Use when the user asks what time it is.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_date",
        "description": "Get the current date and day of the week. Use when the user asks what day or date it is.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_time_in_location",
        "description": "Get the current time in a specific location or timezone. Use when the user asks what time it is in a city, country, or timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city, country, or timezone name (e.g., 'Tokyo', 'Bali', 'New York', 'UTC', 'PST', 'mountain')"
                }
            },
            "required": ["location"]
        }
    },
]


def get_tool_definitions() -> str:
    """Get tool definitions formatted for LFM2-Tool system prompt.

    Returns:
        String with tools wrapped in special tokens for LFM2-Tool
    """
    return f"<|tool_list_start|>{json.dumps(PTZ_TOOLS)}<|tool_list_end|>"


def get_tool_system_prompt(include_instructions: bool = True) -> str:
    """Get complete system prompt section for tools.

    Args:
        include_instructions: Whether to include usage instructions

    Returns:
        Complete tool section for system prompt
    """
    prompt = get_tool_definitions()

    if include_instructions:
        prompt += """

You have access to camera controls. When the user asks you to look around,
find something, or adjust the view, use the appropriate tool.

After using a camera control tool, describe what you see in the new view."""

    return prompt
