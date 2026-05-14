"""
pynlptool - 基于隐马尔可夫模型的中文分词和序列标注库

A Hidden Markov Model (HMM) based Chinese word segmentation and sequence labeling library.
"""

from pathlib import Path
from typing import Literal, Optional, List, Tuple

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

__version__ = "0.3.0"
__author__ = "Luck_mx"
__email__ = "muxinglucky@gmail.com"

# Cache for pretrained model
_model: Optional[HMM] = None
_baseline_model: Optional[HMM] = None

OutputFormat = Literal["tags", "words", "pos", "both"]


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


def _get_bmm_model_path() -> Path:
    """Get the preferred builtin BMM model path."""
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]

    candidates = [
        package_dir / "hmm_bmm.pkl",
        project_root / "models" / "hmm_bmm.pkl",
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


def _render_prediction_output(
    chars: List[str],
    tags: List[str],
    words: List[str],
    words_pos: List[Tuple[str, str]],
    output_format: OutputFormat = "both",
) -> str:
    """Render prediction output in the same layout as the CLI."""
    if output_format == "cut":
        output_format = "words"

    if output_format not in {"tags", "words", "pos", "both"}:
        raise ValueError(f"不支持的输出格式: {output_format}")

    if not chars:
        return ""

    if output_format == "tags":
        return "\n".join(f"{char}\t{tag}" for char, tag in zip(chars, tags))
    if output_format == "words":
        return " ".join(words)
    if output_format == "pos":
        return " | ".join(f"{word}/{pos}" for word, pos in words_pos)

    tag_block = "\n".join(f"{char}\t{tag}" for char, tag in zip(chars, tags))
    word_block = " ".join(words)
    pos_block = " | ".join(f"{word}/{pos}" for word, pos in words_pos)
    return (
        "=== 标签序列 ===\n"
        f"{tag_block}\n\n"
        "=== 分词结果 ===\n"
        f"{word_block}\n\n"
        "=== 词性结果 ===\n"
        f"{pos_block}"
    )


def _predict_components(
    text: str,
    model: Optional[HMM] = None,
    disagreement_threshold: float = 0.32,
    min_avg_margin: float = 0.006,
    max_avg_score_drop: float = 0.08,
    use_fallback: bool = True,
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]]]:
    """Return shared prediction components for the given text.

    When ``model`` is provided, the function uses that model directly.
    Otherwise it mirrors the CLI behavior: prefer the bundled baseline-vs-BMM
    fallback path when both bundled models are available, and otherwise use the
    preferred builtin model.
    """
    text = text.strip()
    if not text:
        return [], [], [], []

    chars = list(text)

    if model is not None:
        normalized = norm_seq(chars)
        tags = model.decode(normalized)
        words = model.cut(text)
        words_pos = tags_to_words_pos(chars, tags)
        return chars, tags, words, words_pos

    if use_fallback:
        baseline_path = _get_baseline_model_path()
        bmm_path = _get_bmm_model_path()
        if baseline_path.exists() and bmm_path.exists():
            baseline_model = HMM.load(str(baseline_path))
            bmm_model = HMM.load(str(bmm_path))
            result = infer_with_fallback(
                baseline_model,
                bmm_model,
                text,
                disagreement_threshold=disagreement_threshold,
                min_avg_margin=min_avg_margin,
                max_avg_score_drop=max_avg_score_drop,
            )
            return chars, result.chosen_tags, result.words, result.words_pos

    builtin_model = load_model()
    normalized = norm_seq(chars)
    tags = builtin_model.decode(normalized)
    words = builtin_model.cut(text)
    words_pos = tags_to_words_pos(chars, tags)
    return chars, tags, words, words_pos


def tags(
    text: str,
    model: Optional[HMM] = None,
    disagreement_threshold: float = 0.32,
    min_avg_margin: float = 0.006,
    max_avg_score_drop: float = 0.08,
    use_fallback: bool = True,
) -> str:
    """Return CLI-style tag output, equivalent to `-o tags`."""
    chars, tags, words, words_pos = _predict_components(
        text,
        model=model,
        disagreement_threshold=disagreement_threshold,
        min_avg_margin=min_avg_margin,
        max_avg_score_drop=max_avg_score_drop,
        use_fallback=use_fallback,
    )
    return _render_prediction_output(chars, tags, words, words_pos, "tags")


def cut(
    text: str,
    model: Optional[HMM] = None,
) -> List[str]:
    """Return the word segmentation result as a list of words."""
    if model is None:
        model = load_model()
    return model.cut(text)


def pos(
    text: str,
    model: Optional[HMM] = None,
    disagreement_threshold: float = 0.32,
    min_avg_margin: float = 0.006,
    max_avg_score_drop: float = 0.08,
    use_fallback: bool = True,
) -> str:
    """Return CLI-style POS output, equivalent to `-o pos`."""
    chars, tags, words, words_pos = _predict_components(
        text,
        model=model,
        disagreement_threshold=disagreement_threshold,
        min_avg_margin=min_avg_margin,
        max_avg_score_drop=max_avg_score_drop,
        use_fallback=use_fallback,
    )
    return _render_prediction_output(chars, tags, words, words_pos, "pos")


def both(
    text: str,
    model: Optional[HMM] = None,
    disagreement_threshold: float = 0.32,
    min_avg_margin: float = 0.006,
    max_avg_score_drop: float = 0.08,
    use_fallback: bool = True,
) -> str:
    """Return full CLI-style output, equivalent to `-o both`."""
    chars, tags, words, words_pos = _predict_components(
        text,
        model=model,
        disagreement_threshold=disagreement_threshold,
        min_avg_margin=min_avg_margin,
        max_avg_score_drop=max_avg_score_drop,
        use_fallback=use_fallback,
    )
    return _render_prediction_output(chars, tags, words, words_pos, "both")


__all__ = [
    # 核心模型
    "HMM",
    "train",
    # 便捷函数
    "load_model",
    "load_baseline_model",
    "cut",
    "tags",
    "pos",
    "both",
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
