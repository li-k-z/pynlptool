"""Fallback inference helpers for baseline-vs-BMM decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pynlptool.data_utils import norm_seq
from pynlptool.model import HMM


def split_tag(tag: str) -> Tuple[str, str]:
    if not tag:
        return "S", "x"
    prefix = tag[0]
    if prefix not in {"B", "M", "E", "S"}:
        return "S", "x"
    if len(tag) == 1:
        return prefix, "x"
    if tag[1] in {"_", "-", "/"}:
        return prefix, tag[2:] or "x"
    return prefix, tag[1:] or "x"


def tag_prefix(tag: str) -> str:
    prefix, _ = split_tag(tag)
    return prefix


def tag_pos(tag: str) -> str:
    _, pos = split_tag(tag)
    return pos


def tags_to_words(chars: Sequence[str], tags: Sequence[str]) -> List[str]:
    words: List[str] = []
    current = ""
    for ch, tag in zip(chars, tags):
        p = tag_prefix(tag)
        if p == "S":
            if current:
                words.append(current)
                current = ""
            words.append(ch)
        elif p == "B":
            if current:
                words.append(current)
            current = ch
        elif p == "M":
            current = (current + ch) if current else ch
        else:
            current = (current + ch) if current else ch
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words


def tags_to_words_pos(chars: Sequence[str], tags: Sequence[str]) -> List[Tuple[str, str]]:
    words_pos: List[Tuple[str, str]] = []
    current_word = ""
    current_pos = "x"

    for ch, tag in zip(chars, tags):
        p = tag_prefix(tag)
        pos = tag_pos(tag)
        if p == "S":
            if current_word:
                words_pos.append((current_word, current_pos))
                current_word = ""
            words_pos.append((ch, pos))
        elif p == "B":
            if current_word:
                words_pos.append((current_word, current_pos))
            current_word = ch
            current_pos = pos
        elif p == "M":
            if not current_word:
                current_word = ch
                current_pos = pos
            else:
                current_word += ch
        else:
            if not current_word:
                current_word = ch
                current_pos = pos
            else:
                current_word += ch
            words_pos.append((current_word, current_pos))
            current_word = ""

    if current_word:
        words_pos.append((current_word, current_pos))

    return words_pos


def word_spans(words: Sequence[str]) -> Set[Tuple[int, int]]:
    spans: Set[Tuple[int, int]] = set()
    cursor = 0
    for word in words:
        end = cursor + len(word)
        spans.add((cursor, end))
        cursor = end
    return spans


def disagreement_ratio(chars: Sequence[str], tags_a: Sequence[str], tags_b: Sequence[str]) -> float:
    words_a = tags_to_words(chars, tags_a)
    words_b = tags_to_words(chars, tags_b)
    span_a = word_spans(words_a)
    span_b = word_spans(words_b)
    union = span_a | span_b
    if not union:
        return 0.0
    return 1.0 - (len(span_a & span_b) / len(union))


def should_fallback(
    disagreement: float,
    bmm_confidence: Dict[str, float],
    baseline_confidence: Dict[str, float],
    disagreement_threshold: float,
    min_avg_margin: float,
    max_avg_score_drop: float,
) -> bool:
    low_confidence = (
        bmm_confidence.get("avg_margin", 0.0) < min_avg_margin
        or (baseline_confidence.get("avg_score", -1e9) - bmm_confidence.get("avg_score", -1e9)) > max_avg_score_drop
    )
    return disagreement >= disagreement_threshold and low_confidence


@dataclass
class FallbackResult:
    chosen_model: str
    words: List[str]
    words_pos: List[Tuple[str, str]]
    baseline_tags: List[str]
    bmm_tags: List[str]
    chosen_tags: List[str]
    baseline_confidence: Dict[str, float]
    bmm_confidence: Dict[str, float]
    disagreement: float


def infer_with_fallback(
    baseline_model: HMM,
    bmm_model: HMM,
    text: str,
    disagreement_threshold: float = 0.32,
    min_avg_margin: float = 0.006,
    max_avg_score_drop: float = 0.08,
) -> FallbackResult:
    chars = list(text)
    obs = norm_seq(chars)

    baseline_tags, baseline_confidence = baseline_model.decode_with_confidence(obs)
    bmm_tags, bmm_confidence = bmm_model.decode_with_confidence(obs)
    disagreement = disagreement_ratio(chars, baseline_tags, bmm_tags)

    use_baseline = should_fallback(
        disagreement,
        bmm_confidence,
        baseline_confidence,
        disagreement_threshold,
        min_avg_margin,
        max_avg_score_drop,
    )

    chosen_model = "baseline" if use_baseline else "bmm"
    chosen_tags = baseline_tags if use_baseline else bmm_tags
    words = tags_to_words(chars, chosen_tags)
    words_pos = tags_to_words_pos(chars, chosen_tags)

    return FallbackResult(
        chosen_model=chosen_model,
        words=words,
        words_pos=words_pos,
        baseline_tags=baseline_tags,
        bmm_tags=bmm_tags,
        chosen_tags=chosen_tags,
        baseline_confidence=baseline_confidence,
        bmm_confidence=bmm_confidence,
        disagreement=disagreement,
    )