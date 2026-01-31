"""
数据加载和预处理工具
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


def norm_char(ch: str) -> str:
    """
    字符归一化，减少数据稀疏性。

    Args:
        ch: 输入字符

    Returns:
        归一化后的字符或特殊标记
    """
    if ch.isdigit():
        return "<NUM>"
    # 常见中文数字
    if ch in "零〇一二三四五六七八九十百千万亿":
        return "<CNUM>"
    # 英文字母（不区分大小写）
    if "a" <= ch.lower() <= "z":
        return "<LAT>"
    return ch


def norm_seq(seq: List[str]) -> List[str]:
    """
    序列归一化。

    Args:
        seq: 字符列表

    Returns:
        归一化后的字符列表
    """
    return [norm_char(c) for c in seq]


@dataclass
class Sentence:
    """
    带标注的句子。

    Attributes:
        observations: 观察符号列表（如字符）
        tags: 对应的标签列表
    """

    observations: List[str]
    tags: List[str]


def _read_text(path: str) -> str:
    """读取文本文件，自动尝试多种编码。"""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_data(path: str) -> List[Sentence]:
    """
    从文件加载标注数据。

    文件格式：每行一个"字符 标签"对，句子之间用空行分隔。

    Args:
        path: 数据文件路径

    Returns:
        Sentence 对象列表
    """
    text = _read_text(path)
    sentences: List[Sentence] = []
    obs: List[str] = []
    tags: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if obs:
                sentences.append(Sentence(observations=obs, tags=tags))
                obs, tags = [], []
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        char = norm_char(parts[0])
        tag = parts[1]
        obs.append(char)
        tags.append(tag)

    if obs:
        sentences.append(Sentence(observations=obs, tags=tags))

    return sentences
