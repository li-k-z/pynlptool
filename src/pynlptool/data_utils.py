"""
数据加载和预处理工具
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple


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


def tags_to_words(chars: Sequence[str], tags: Sequence[str]) -> List[str]:
    """将字符序列与 BMES 标签转换为词序列。"""
    words: List[str] = []
    current = ""
    for ch, tag in zip(chars, tags):
        p = tag[0] if tag else "S"
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
        else:  # E
            current = (current + ch) if current else ch
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words


def build_dictionary_from_sequences(
    sequences: Iterable[Tuple[Sequence[str], Sequence[str]]],
    min_word_len: int = 2,
) -> Set[str]:
    """从标注序列构建词典（用于 BMM 特征）。"""
    lexicon: Set[str] = set()
    for obs, tags in sequences:
        words = tags_to_words(obs, tags)
        for w in words:
            if len(w) >= min_word_len:
                lexicon.add(w)
    return lexicon


def load_lexicon(path: str, normalize: bool = True) -> Set[str]:
    """从词典文件加载词条，兼容 `词 频次 词性` 格式。"""
    text = _read_text(path)
    lexicon: Set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        word = line.split()[0]
        if normalize:
            word = "".join(norm_seq(list(word)))
        if word:
            lexicon.add(word)
    return lexicon


def _fmm(tokens: Sequence[str], lexicon: Set[str], max_word_len: int) -> List[str]:
    """正向最大匹配。"""
    n = len(tokens)
    i = 0
    words: List[str] = []
    while i < n:
        take = tokens[i]
        upper = min(max_word_len, n - i)
        for L in range(upper, 0, -1):
            cand = "".join(tokens[i : i + L])
            if L == 1 or cand in lexicon:
                take = cand
                break
        words.append(take)
        i += len(take)
    return words


def _rmm(tokens: Sequence[str], lexicon: Set[str], max_word_len: int) -> List[str]:
    """逆向最大匹配。"""
    j = len(tokens)
    words_rev: List[str] = []
    while j > 0:
        take = tokens[j - 1]
        upper = min(max_word_len, j)
        for L in range(upper, 0, -1):
            cand = "".join(tokens[j - L : j])
            if L == 1 or cand in lexicon:
                take = cand
                break
        words_rev.append(take)
        j -= len(take)
    words_rev.reverse()
    return words_rev


def bmm_segment(tokens: Sequence[str], lexicon: Set[str], max_word_len: int = 6) -> List[str]:
    """双向最大匹配分词。"""
    if not tokens:
        return []
    if not lexicon:
        return list(tokens)

    f = _fmm(tokens, lexicon, max_word_len=max_word_len)
    r = _rmm(tokens, lexicon, max_word_len=max_word_len)

    if len(f) != len(r):
        return f if len(f) < len(r) else r

    f_single = sum(1 for w in f if len(w) == 1)
    r_single = sum(1 for w in r if len(w) == 1)
    if f_single != r_single:
        return f if f_single < r_single else r

    return f


def words_to_bmes(words: Sequence[str]) -> List[str]:
    """词序列转换为字级 BMES 序列。"""
    tags: List[str] = []
    for w in words:
        if not w:
            continue
        if len(w) == 1:
            tags.append("S")
        elif len(w) == 2:
            tags.extend(["B", "E"])
        else:
            tags.append("B")
            tags.extend(["M"] * (len(w) - 2))
            tags.append("E")
    return tags


def bmm_tags(tokens: Sequence[str], lexicon: Set[str], max_word_len: int = 6) -> List[str]:
    """根据 BMM 词典分词输出字级 BMES 标签序列。"""
    words = bmm_segment(tokens, lexicon=lexicon, max_word_len=max_word_len)
    tags = words_to_bmes(words)
    if len(tags) != len(tokens):
        # 理论上不会发生；作为保护回退到逐字切分。
        return ["S"] * len(tokens)
    return tags


def augment_observations_with_bmm(
    tokens: Sequence[str],
    lexicon: Set[str],
    joiner: str = "|",
    max_word_len: int = 6,
) -> List[str]:
    """
    使用 +0 对位策略，把 `字` 与 `BMM字标签` 拼接为新观测。

    输出形如: `今|B`, `天|E`。
    """
    tags = bmm_tags(tokens, lexicon=lexicon, max_word_len=max_word_len)
    return [f"{tok}{joiner}{tag}" for tok, tag in zip(tokens, tags)]


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
