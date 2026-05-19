def detect_error_type(response, expected, context):
    response_lower = response.lower()
    expected_lower = expected.lower()

    # -----------------------
    # System error
    # -----------------------
    if "error" in response_lower:
        return "system_error"

    # -----------------------
    # Over fallback
    # -----------------------
    if "context does not contain" in response_lower:
        if expected_lower not in response_lower:
            return "over_fallback"

    # -----------------------
    # Exact match
    # -----------------------
    if expected_lower in response_lower:
        return "correct"

    # -----------------------
    # Partial match
    # -----------------------
    overlap = set(expected_lower.split()) & set(response_lower.split())
    if len(overlap) > 2:
        return "partial_correct"

    # -----------------------
    # Default → hallucination
    # -----------------------
    return "hallucination"


def is_verbose(response):
    sentences = [s for s in response.split(".") if s.strip()]
    return len(sentences) > 2