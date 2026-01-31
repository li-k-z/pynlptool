"""
模型评估工具
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


def evaluate(
    sequences: Iterable[Tuple[List[str], List[str]]],
    predictions: Iterable[List[str]],
) -> Dict[str, float]:
    """
    评估模型预测结果。

    Args:
        sequences: (观察序列, 标准标签) 元组的可迭代对象
        predictions: 预测标签序列的可迭代对象

    Returns:
        评估指标字典：
        - accuracy: 整体准确率
        - macro_precision: 宏平均精确率
        - macro_recall: 宏平均召回率
        - macro_f1: 宏平均F1分数
        - token_count: 总token数
        - tag_count: 标签类别数
    """
    total = 0
    correct = 0

    gold_counts: Counter = Counter()
    pred_counts: Counter = Counter()
    correct_counts: Counter = Counter()

    for (obs, gold_tags), pred_tags in zip(sequences, predictions):
        for g, p in zip(gold_tags, pred_tags):
            total += 1
            if g == p:
                correct += 1
                correct_counts[g] += 1
            gold_counts[g] += 1
            pred_counts[p] += 1

    tags = sorted(set(gold_counts.keys()) | set(pred_counts.keys()))
    macro_f1 = 0.0
    macro_p = 0.0
    macro_r = 0.0
    for t in tags:
        tp = correct_counts[t]
        p = tp / pred_counts[t] if pred_counts[t] > 0 else 0.0
        r = tp / gold_counts[t] if gold_counts[t] > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        macro_p += p
        macro_r += r
        macro_f1 += f1

    n_tags = max(len(tags), 1)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p / n_tags,
        "macro_recall": macro_r / n_tags,
        "macro_f1": macro_f1 / n_tags,
        "token_count": float(total),
        "tag_count": float(n_tags),
    }


def report(metrics: Dict[str, float]) -> str:
    """
    将评估指标格式化为可读报告。

    Args:
        metrics: evaluate() 返回的指标字典

    Returns:
        格式化的报告字符串
    """
    lines = [
        "HMM 模型评估报告",
        "=" * 30,
        f"Token 数量: {int(metrics['token_count'])}",
        f"标签类别数: {int(metrics['tag_count'])}",
        f"准确率: {metrics['accuracy']:.4f}",
        f"宏平均精确率: {metrics['macro_precision']:.4f}",
        f"宏平均召回率: {metrics['macro_recall']:.4f}",
        f"宏平均 F1: {metrics['macro_f1']:.4f}",
    ]
    return "\n".join(lines) + "\n"
