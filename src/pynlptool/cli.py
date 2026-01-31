"""
pynlptool 命令行界面
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from pynlptool.model import HMM
from pynlptool.data_utils import norm_seq


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
        choices=["tags", "words", "both"],
        default="both",
        help="输出格式: tags, words, 或 both（默认: both）",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
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
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"错误: 模型文件未找到: {args.model}", file=sys.stderr)
            sys.exit(1)
        model = HMM.load(str(model_path))
    else:
        # 优先使用内置模型
        from pynlptool import load_model as _load_builtin
        try:
            model = _load_builtin()
        except FileNotFoundError:
            # 回退到外部路径
            possible_paths = [
                Path.cwd() / "models" / "model.pkl",
                Path.cwd() / "model.pkl",
                Path.home() / ".pynlptool" / "model.pkl",
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
    
    # 处理文本
    chars = list(text)
    normalized = norm_seq(chars)
    tags = model.decode(normalized)
    words = model.cut(text)
    
    if args.output_format == "tags":
        for char, tag in zip(chars, tags):
            print(f"{char}\t{tag}")
    elif args.output_format == "words":
        print(" ".join(words))
    else:  # both
        print("=== 标签序列 ===")
        for char, tag in zip(chars, tags):
            print(f"{char}\t{tag}")
        print("\n=== 分词结果 ===")
        print(" ".join(words))


if __name__ == "__main__":
    main()
