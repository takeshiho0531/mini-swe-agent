"""Bound diagnostic output without changing the observation sent to the model."""

import time

RAW_OUTPUT_MAX_CHARS = 64 * 1024
_OMITTED = "\n[raw output omitted from metadata]\n"


def observation_metadata(output: dict) -> dict:
    """Retain at most 64 Ki characters of raw output, including a gap marker.

    Call after rendering the observation from the original output. Only extra
    metadata is shortened; API input and submission text remain unchanged.
    Keep both ends and record the original length so this cannot be mistaken
    for a complete transcript. Small outputs keep the existing schema.
    """
    extra = {
        "raw_output": output.get("output", ""),
        "returncode": output.get("returncode"),
        "timestamp": time.time(),
        "exception_info": output.get("exception_info"),
        **output.get("extra", {}),
    }
    raw = extra["raw_output"]
    if isinstance(raw, str) and len(raw) > RAW_OUTPUT_MAX_CHARS:
        head = (RAW_OUTPUT_MAX_CHARS - len(_OMITTED)) // 2
        tail = RAW_OUTPUT_MAX_CHARS - len(_OMITTED) - head
        extra.update(
            raw_output=raw[:head] + _OMITTED + raw[-tail:],
            raw_output_original_chars=len(raw),
            raw_output_truncated=True,
        )
    return extra
