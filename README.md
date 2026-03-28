# pynlptool

[![PyPI version](https://badge.fury.io/py/pynlptool.svg)](https://badge.fury.io/py/pynlptool)
[![Python Version](https://img.shields.io/pypi/pyversions/pynlptool.svg)](https://pypi.org/project/pynlptool/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于隐马尔可夫模型（HMM）的中文分词和序列标注库。

A Hidden Markov Model (HMM) based Chinese word segmentation and sequence labeling library.

## 特性 / Features

- 纯 Python 实现，无外部依赖
- **内置预训练模型，开箱即用**
- 支持 BMES 标注体系
- 支持自定义训练数据
- 内置模型评估和超参数调优
- 模型序列化保存与加载

## 安装 / Installation

```bash
pip install pynlptool
```

## 快速开始 / Quick Start

### 使用内置模型（推荐）

```python
from pynlptool import cut, load_model

# 方式一：直接分词（最简单）
words = cut("今天天气不错")
print(words)  # ['今天', '天气', '不错']

# 方式二：加载模型后多次使用
model = load_model()
words1 = model.cut("今天天气不错")
words2 = model.cut("我喜欢编程")

# 获取标注序列
tags = model.decode(list("今天天气"))
print(list(zip(list("今天天气"), tags)))
```

### 训练模型

```python
from pynlptool import train, HMM

# 准备训练数据: 列表，每个元素为 (observations, tags) 元组
sequences = [
    (["今", "天", "天", "气", "不", "错"], ["B", "E", "B", "E", "B", "E"]),
    (["我", "喜", "欢", "编", "程"], ["S", "B", "E", "B", "E"]),
]

# 训练模型
model = train(sequences, alpha=0.5, min_freq=1)

# 保存模型
model.save("my_model.pkl")
```

### 加载模型并预测

```python
from pynlptool import HMM

# 加载模型
model = HMM.load("my_model.pkl")

# 预测
text = list("今天天气真好")
tags = model.decode(text)
print(list(zip(text, tags)))
```

### 使用数据工具

```python
from pynlptool import load_data, norm_char, norm_seq

# 加载标注数据
sentences = load_data("train.txt")

# 字符归一化
normalized = norm_seq(list("2024年"))
print(normalized)  # ['<NUM>', '<NUM>', '<NUM>', '<NUM>', '年']
```

### 模型评估

```python
from pynlptool import load_data, evaluate, report

# 评估模型
test_sentences = load_data("test.txt")
test_sequences = [(s.observations, s.tags) for s in test_sentences]
predictions = [model.decode(obs) for obs, _ in test_sequences]
metrics = evaluate(test_sequences, predictions)

# 生成报告
r = report(metrics)
print(r)
```

## 命令行工具 / CLI

安装后可以使用命令行工具进行预测：

```bash
pynlptool "今天天气不错"
```

更多 CLI 用法：

```bash
# 只输出分词结果
pynlptool "今天天气不错" -o words

# 只输出字符标签
pynlptool "今天天气不错" -o tags

# 指定自定义模型
pynlptool "今天天气不错" -m my_model.pkl

# 从标准输入读取
echo 今天天气不错 | pynlptool -o both
```

参数说明：

- `-o, --output-format`: `tags` / `words` / `both`（默认 `both`）
- `-m, --model`: 指定模型文件路径（pickle）
- `--version`: 查看版本

## 边界行为说明 / Edge Cases

根据当前实现与功能测试，以下行为是确定的：

- `cut("")` 返回 `[]`
- 单字符输入会返回单元素词列表
- 数字、英文会先进行归一化后再解码，但输出仍保留原始字符
- 中英数混合输入可直接处理（例如 `Hello世界2026`）
- `tag(text)` 返回 `(原始字符, 标签)` 二元组列表

## 数据格式 / Data Format

训练数据应为空行分隔的句子，每行一个字符及其标签：

```
今 B
天 E
天 B
气 E

我 S
喜 B
欢 E
```

## API 参考 / API Reference

### `load_model()`

加载内置的预训练模型，可直接用于分词。模型会被缓存，多次调用不会重复加载。

### `cut(text)`

使用内置模型对文本进行分词的便捷函数。

- `text`: 待分词的中文文本

### `tag(text)`

使用内置模型获取字符标注结果。

- `text`: 待标注的中文文本
- 返回: `List[Tuple[str, str]]` 字符-标签对列表

### `show(text)`

使用内置模型获取格式化的标注显示结果。

- `text`: 待标注的中文文本
- 返回: 格式化的字符串

### `train(sequences, alpha=0.1, min_freq=3, unk_token="<UNK>", tag_penalty=-20.0)`

训练 HMM 模型。

- `sequences`: 训练数据，`Iterable[Tuple[List[str], List[str]]]`
- `alpha`: 平滑参数
- `min_freq`: 最小词频阈值
- `unk_token`: 未知词标记
- `tag_penalty`: 标签惩罚系数

### `HMM`

HMM 模型类。

- `decode(observations)`: Viterbi 解码，返回标签序列
- `cut(text)`: 对中文文本进行分词，返回词列表
- `save(path)`: 保存模型
- `load(path)`: 加载模型（静态方法）

### `load_data(path)`

加载标注数据文件。

### `evaluate(sequences, predictions)`

评估模型预测结果。

- 返回指标包括: `accuracy`、`macro_precision`、`macro_recall`、`macro_f1`、`token_count`、`tag_count`

### `norm_char(ch)` / `norm_seq(seq)`

字符/序列归一化处理。

## 许可证 / License

MIT License
