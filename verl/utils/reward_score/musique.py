import re
import string
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).strip()


def extract_answer(s: str) -> str:
    last_open_tag_pos = s.rfind("<answer>")
    if last_open_tag_pos == -1:
        return ""

    first_close_tag_pos = s.find("</answer>", last_open_tag_pos)
    if first_close_tag_pos == -1:
        return ""

    start = last_open_tag_pos + len("<answer>")
    end = first_close_tag_pos
    return s[start:end].strip()


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def f1_score(prediction: str, references: List[str]) -> float:
    if not prediction:
        return 0.0
    max_f1 = 0.0
    prediction_tokens = prediction.split()
    for reference in references:
        reference_tokens = reference.split()
        if not reference_tokens:
            continue
        common_tokens = Counter(prediction_tokens) & Counter(reference_tokens)
        num_common = sum(common_tokens.values())
        if num_common == 0:
            continue
        precision = num_common / len(prediction_tokens)
        recall = num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1


# def acc_reward(predict_str: str, references: List[str]) -> float:
#     prediction = normalize_answer(extract_answer(predict_str))
#     references = [normalize_answer(ref) for ref in references]

#     return f1_score(prediction, references)


def acc_reward(predict_str: str, references: List[str]) -> float:
    prediction = extract_answer(predict_str)

    return f1_score(prediction, references)


def compute_score(
    predict_str, ground_truth, answer_aliases, format_score=0.0, score=1.0
):
    references = [ground_truth]
    if len(answer_aliases) != 0:
        references.extend(answer_aliases)

    return (1.0 - format_score) * acc_reward(
        predict_str, references
    ) + format_score * format_reward(predict_str)
