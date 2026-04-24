"""
pynlptool 命令行界面
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from pynlptool import HMM, infer_with_fallback, tag_pos, tags_to_words_pos
from pynlptool.data_utils import norm_seq


def _cli_version() -> str:
    """Return installed package version for CLI display."""
    from pynlptool import __version__

    return __version__


def predict_sentence(model: HMM, sentence: str) -> None:
    """打印句子的预测结果。"""
    raw_tokens = list(sentence.strip())
    if not raw_tokens:
        print("输入为空，无法预测。")
        return
    
    tokens = norm_seq(raw_tokens)
    tags = model.decode(tokens)
    
    print("字符\t标签")
    print("-" * 20)
    for tok_raw, tag in zip(raw_tokens, tags):
        print(f"{tok_raw}\t{tag}")
    
    # 同时显示分词结果
    words = model.cut(sentence)
    print("\n分词结果:")
    print(" / ".join(words))


def predict_sentence_with_fallback(baseline_model: HMM, bmm_model: HMM, sentence: str) -> None:
    """打印回退推理后的标签、分词和词性结果。"""
    text = sentence.strip()
    if not text:
        print("输入为空，无法预测。")
        return

    result = infer_with_fallback(baseline_model, bmm_model, text)
    chars = list(text)

    print(f"回退策略: {result.chosen_model} (disagree={result.disagreement:.4f})")
    print(
        "baseline avg_score={:.4f}, avg_margin={:.4f} | bmm avg_score={:.4f}, avg_margin={:.4f}".format(
            result.baseline_confidence.get("avg_score", 0.0),
            result.baseline_confidence.get("avg_margin", 0.0),
            result.bmm_confidence.get("avg_score", 0.0),
            result.bmm_confidence.get("avg_margin", 0.0),
        )
    )
    print("字符\t标签")
    print("-" * 20)
    for tok_raw, tag in zip(chars, result.chosen_tags):
        print(f"{tok_raw}\t{tag}")

    print("\n分词结果:")
    print(" / ".join(result.words))

    print("\n词性结果:")
    print(" | ".join(f"{word}/{pos}" for word, pos in result.words_pos))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于HMM的中文分词和序列标注工具。",
        prog="pynlptool",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="要分词的中文文本（如不提供则从标准输入读取）",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="模型文件路径（pickle格式）",
    )
    parser.add_argument(
        "-o", "--output-format",
        choices=["tags", "words", "pos", "both"],
        default="both",
        help="输出格式: tags, words, pos, 或 both（默认: both）",
    )
    parser.add_argument(
        "--disagreement-threshold",
        type=float,
        default=0.32,
        help="启用回退时的分歧阈值（默认: 0.32）",
    )
    parser.add_argument(
        "--min-avg-margin",
        type=float,
        default=0.006,
        help="启用回退时 BMM 的最低平均路径分差阈值（默认: 0.006）",
    )
    parser.add_argument(
        "--max-avg-score-drop",
        type=float,
        default=0.08,
        help="启用回退时 BMM 相对基线的最大平均分差（默认: 0.08）",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_cli_version()}",
    )
    return parser.parse_args()


def main() -> None:
    """CLI主入口。"""
    args = parse_args()
    
    # 获取输入文本
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()
    
    if not text:
        print("错误: 未提供输入文本。", file=sys.stderr)
        sys.exit(1)
    
    # 加载模型
    use_fallback = args.model is None

    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"错误: 模型文件未找到: {args.model}", file=sys.stderr)
            sys.exit(1)
        model = HMM.load(str(model_path))
    else:
        from pynlptool import load_model as _load_builtin

        try:
            model = _load_builtin()
        except FileNotFoundError:
            possible_paths = [
                Path.cwd() / "models" / "hmm_bmm.pkl",
                Path.cwd() / "models" / "model.pkl",
                Path.cwd() / "model.pkl",
                Path.cwd() / "hmm_bmm.pkl",
                Path.home() / ".pynlptool" / "model.pkl",
                Path.home() / ".pynlptool" / "hmm_bmm.pkl",
            ]
            model_path = None
            for p in possible_paths:
                if p.exists():
                    model_path = p
                    break

            if model_path is None:
                print(
                    "错误: 未找到模型文件。"
                    "请使用 -m/--model 选项指定模型路径。",
                    file=sys.stderr,
                )
                sys.exit(1)
            model = HMM.load(str(model_path))
            use_fallback = False

    baseline_model = None
    bmm_model = None
    if use_fallback:
        package_dir = Path(__file__).resolve().parent
        baseline_candidates = [
            package_dir / "model.pkl",
            ROOT / "models" / "model.pkl",
        ]
        bmm_candidates = [
            package_dir / "hmm_bmm.pkl",
            ROOT / "models" / "hmm_bmm.pkl",
        ]
        baseline_path = next((p for p in baseline_candidates if p.exists()), None)
        bmm_path = next((p for p in bmm_candidates if p.exists()), None)
        if baseline_path is not None and bmm_path is not None:
            baseline_model = HMM.load(str(baseline_path))
            bmm_model = HMM.load(str(bmm_path))
        else:
            use_fallback = False
    
    # 处理文本
    chars = list(text)
    normalized = norm_seq(chars)

    if use_fallback and baseline_model is not None and bmm_model is not None:
        fallback_result = infer_with_fallback(
            baseline_model,
            bmm_model,
            text,
            disagreement_threshold=args.disagreement_threshold,
            min_avg_margin=args.min_avg_margin,
            max_avg_score_drop=args.max_avg_score_drop,
        )
        tags = fallback_result.chosen_tags
        words = fallback_result.words
        words_pos = fallback_result.words_pos
    else:
        tags = model.decode(normalized)
        words = model.cut(text)
        words_pos = tags_to_words_pos(chars, tags)
    
    if args.output_format == "tags":
        for char, tag in zip(chars, tags):
            print(f"{char}\t{tag}")
    elif args.output_format == "words":
        print(" ".join(words))
    elif args.output_format == "pos":
        print(" | ".join(f"{word}/{pos}" for word, pos in words_pos))
    else:  # both
        print("=== 标签序列 ===")
        for char, tag in zip(chars, tags):
            print(f"{char}\t{tag}")
        print("\n=== 分词结果 ===")
        print(" ".join(words))
        print("\n=== 词性结果 ===")
        print(" | ".join(f"{word}/{pos}" for word, pos in words_pos))


if __name__ == "__main__":
    main()
