"""
Function calling schemas, API payload formatting, response parsing,
and rigorous tool/argument comparison logic.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.dataset import ToolDefinition


def build_gemini_tools_payload(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    """Convert a list of ToolDefinition models into Gemini API tools format."""
    declarations = [t.to_gemini_schema() for t in tools]
    return [{"function_declarations": declarations}]


def parse_gemini_response(response_json: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """
    Extract function call name, parsed arguments, and any error message from raw Gemini API JSON response.
    Returns (predicted_tool, predicted_args, error_message).
    """
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return None, {}, "No candidates returned in API response"

        first_candidate = candidates[0]
        content = first_candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            if "functionCall" in part:
                fn_call = part["functionCall"]
                tool_name = fn_call.get("name")
                args = fn_call.get("args", {})
                return tool_name, args, None

        # If text returned instead of a function call
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        combined_text = " ".join(text_parts).strip()
        if combined_text:
            return None, {}, f"Model returned text instead of functionCall: '{combined_text[:100]}'"

        return None, {}, "No functionCall part found in candidate parts"

    except Exception as e:
        return None, {}, f"Parsing exception: {str(e)}"


def compare_arguments(
    predicted: Dict[str, Any],
    expected: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare predicted tool arguments with ground truth expected arguments.
    Allows numeric tolerance (e.g. 150 == 150.0) and case-insensitive string matching.
    Returns (is_correct, details_dict).
    """
    mismatches = {}

    for exp_k, exp_v in expected.items():
        if exp_k not in predicted:
            mismatches[exp_k] = {
                "status": "missing",
                "expected": exp_v,
                "predicted": None
            }
            continue

        pred_v = predicted[exp_k]

        # Numeric comparison
        if isinstance(exp_v, (int, float)) and isinstance(pred_v, (int, float)):
            if abs(float(exp_v) - float(pred_v)) > 1e-4:
                mismatches[exp_k] = {
                    "status": "value_mismatch",
                    "expected": exp_v,
                    "predicted": pred_v
                }
        # String comparison
        elif isinstance(exp_v, str) and isinstance(pred_v, str):
            if exp_v.strip().lower() != pred_v.strip().lower():
                mismatches[exp_k] = {
                    "status": "string_mismatch",
                    "expected": exp_v,
                    "predicted": pred_v
                }
        # General equality
        else:
            if exp_v != pred_v:
                mismatches[exp_k] = {
                    "status": "general_mismatch",
                    "expected": exp_v,
                    "predicted": pred_v
                }

    is_correct = len(mismatches) == 0
    return is_correct, mismatches
