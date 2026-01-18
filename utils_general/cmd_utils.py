# cmd_utils.py - Lightweight command-line utilities
# No heavy imports (torch, etc.) - safe for quick startup scripts

from typing import Dict


def dict_to_cmd_args(params: Dict, equals=False) -> list[str]:
    """Convert keyword arguments to command-line argument list."""
    parts = []
    for key, value in sorted(params.items()):
        # If the value is a boolean, only include it if True.
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
        elif value is not None:
            if equals:
                parts.append(f"--{key}={value}")
            else:
                parts.append(f"--{key}")
                parts.append(str(value))
    return parts
