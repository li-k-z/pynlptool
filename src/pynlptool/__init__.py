"""
pynlptool - 基于隐马尔可夫模型的中文分词和序列标注库

A Hidden Markov Model (HMM) based Chinese word segmentation and sequence labeling library.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

from pynlptool.model import HMM, train
from pynlptool.data_utils import (
    Sentence,
    load_data,
    norm_char,
    norm_seq,
)
from pynlptool.evaluate import evaluate, report

__version__ = "0.1.0"
__author__ = "Luck_mx"
__email__ = "muxinglucky@gmail.com"

# Cache for pretrained model
_model: Optional[HMM] = None


def _get_model_path() -> Path:
    """Get the path to the bundled pretrained model."""
    return Path(__file__).parent / "model.pkl"


def load_model() -> HMM:
    """
    加载内置的预训练模型。

    模型会被缓存，多次调用不会重复加载。

    Returns:
        HMM: 预训练模型实例

    Example:
        >>> from pynlptool import load_model
        >>> model = load_model()
        >>> words = model.cut("今天天气不错")
    """
    global _model
    if _model is None:
        model_path = _get_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"模型文件未找到: {model_path}. "
                "请重新安装包或训练自己的模型。"
            )
        _model = HMM.load(str(model_path))
    return _model


def cut(text: str) -> List[str]:
    """
    中文分词。

    Args:
        text: 待分词的中文文本

    Returns:
        分词结果列表

    Example:
        >>> from pynlptool import cut
        >>> cut("今天天气不错")
        ['今天', '天气', '不错']
    """
    model = load_model()
    return model.cut(text)


def tag(text: str) -> List[Tuple[str, str]]:
    """
    序列标注，返回字符-标签对。

    Args:
        text: 待标注的中文文本

    Returns:
        (字符, 标签) 元组列表

    Example:
        >>> from pynlptool import tag
        >>> tag("今天")
        [('今', 'B_t'), ('天', 'E_t')]
    """
    model = load_model()
    chars = list(text)
    normalized = norm_seq(chars)
    tags = model.decode(normalized)
    return list(zip(chars, tags))


def show(text: str) -> str:
    """
    格式化显示标注结果。

    Args:
        text: 待标注的中文文本

    Returns:
        格式化的标注结果字符串

    Example:
        >>> from pynlptool import show
        >>> print(show("今天"))
    """
    result = tag(text)
    lines = ["字符\t标签", "-" * 16]
    for char, label in result:
        lines.append(f"{char}\t{label}")
    return "\n".join(lines)


__all__ = [
    # 核心模型
    "HMM",
    "train",
    # 便捷函数
    "load_model",
    "cut",
    "tag",
    "show",
    # 数据工具
    "Sentence",
    "load_data",
    "norm_char",
    "norm_seq",
    # 评估
    "evaluate",
    "report",
    # 元信息
    "__version__",
]
