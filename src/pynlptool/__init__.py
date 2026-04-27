"""
pynlptool - 基于隐马尔可夫模型的中文分词和序列标注库

A Hidden Markov Model (HMM) based Chinese word segmentation and sequence labeling library.
"""

from pathlib import Path
from typing import Optional, List, Tuple

from pynlptool.model import HMM, train
from pynlptool.data_utils import (
    Sentence,
    augment_observations_with_bmm,
    bmm_segment,
    bmm_tags,
    build_dictionary_from_sequences,
    load_data,
    load_lexicon,
    norm_char,
    norm_seq,
)
from pynlptool.evaluate import evaluate, report
from pynlptool.fallback import (
    FallbackResult,
    disagreement_ratio,
    infer_with_fallback,
    should_fallback,
    tag_pos,
    tag_prefix,
    tags_to_words,
    tags_to_words_pos,
    word_spans,
)

__version__ = "0.2.3"
__author__ = "Luck_mx"
__email__ = "muxinglucky@gmail.com"

# Cache for pretrained model
_model: Optional[HMM] = None
_baseline_model: Optional[HMM] = None


def _get_model_path() -> Path:
    """Get the preferred builtin model path.

    Priority:
    1. Bundled package `hmm_bmm.pkl` (new default model)
    2. Bundled package `model.pkl` (legacy fallback)
    3. Project-root `models/hmm_bmm.pkl` (workspace fallback)
    4. Project-root `models/model.pkl` (workspace legacy fallback)
    """
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]

    candidates = [
        package_dir / "hmm_bmm.pkl",
        package_dir / "model.pkl",
        project_root / "models" / "hmm_bmm.pkl",
        project_root / "models" / "model.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


def _get_baseline_model_path() -> Path:
    """Get the preferred builtin baseline model path."""
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]

    candidates = [
        package_dir / "model.pkl",
        project_root / "models" / "model.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


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


def load_baseline_model() -> HMM:
    """
    加载内置的基线模型（model.pkl）。

    Returns:
        HMM: 基线模型实例
    """
    global _baseline_model
    if _baseline_model is None:
        model_path = _get_baseline_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"基线模型文件未找到: {model_path}. "
                "请重新安装包或手动提供 model.pkl。"
            )
        _baseline_model = HMM.load(str(model_path))
    return _baseline_model


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
    "load_baseline_model",
    "cut",
    "tag",
    "show",
    # 数据工具
    "Sentence",
    "bmm_segment",
    "bmm_tags",
    "augment_observations_with_bmm",
    "build_dictionary_from_sequences",
    "load_lexicon",
    "load_data",
    "norm_char",
    "norm_seq",
    # 评估
    "evaluate",
    "report",
    # 回退推理
    "FallbackResult",
    "disagreement_ratio",
    "infer_with_fallback",
    "should_fallback",
    "tag_pos",
    "tag_prefix",
    "tags_to_words",
    "tags_to_words_pos",
    "word_spans",
    # 元信息
    "__version__",
]
