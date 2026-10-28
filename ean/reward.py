import re

from mathruler.grader import extract_boxed_content, grade_answer

FORMAT_PATTERN = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
TAG_SPACING_PATTERN = re.compile(r"\s*(<|>|/)\s*")


def normalize_response(response: str) -> str:
    return TAG_SPACING_PATTERN.sub(r"\1", response)


def accuracy_reward(response: str, ground_truth: str) -> float:
    return float(grade_answer(extract_boxed_content(response), ground_truth))


def format_reward(response: str) -> float:
    return float(FORMAT_PATTERN.fullmatch(response) is not None)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_weight: float = 0.1) -> dict:
    response = normalize_response(solution_str)
    accuracy = accuracy_reward(response, "" if ground_truth is None else str(ground_truth))
    format_score = format_reward(response)
    return {
        "score": (1 - format_weight) * accuracy + format_weight * format_score,
        "accuracy": accuracy,
        "format": format_score,
    }
