# pynlptool

当前版本: `0.2.3`

[![PyPI version](https://badge.fury.io/py/pynlptool.svg)](https://badge.fury.io/py/pynlptool)
[![Python Version](https://img.shields.io/pypi/pyversions/pynlptool.svg)](https://pypi.org/project/pynlptool/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于隐马尔可夫模型（HMM）的中文分词与序列标注工具库。

A lightweight HMM-based library for Chinese word segmentation and sequence labeling.

## 亮点

- 纯Python实现，安装简单
- 内置BMM增强预训练模型，开箱即用
- 支持基线模型与BMM-HMM动态回退推理
- 支持BMES混合标注与自定义训练
- 支持模型保存/加载与离线评估
- 提供Python API与CLI两种使用方式

## 安装

```bash
pip install pynlptool
```

## 快速开始

先用 Python 直接上手，再用 CLI 做同样的事情。下面两种方式对应的是同一套输出逻辑，用户可以按自己的使用场景选择其一。

### 1) Python 直接调用

```python
from pynlptool import cut, tags, pos, both

text = "南京长江大桥是长江上第一座由中国自主设计、自行建造的双层式铁路、公路两用桥梁，是新中国的标志性工程与时代精神象征，被誉为 “争气桥”。"

print(tags(text))
print(cut(text))
print(pos(text))
print(both(text))
```

`cut(text)` 返回分词列表，最适合直接在 Python 里继续处理；`tags(text)`、`pos(text)`、`both(text)` 与 CLI 的 `-o tags / pos / both` 一一对应，返回格式也完全一致。`load_model()` 适合需要直接操作模型对象的场景。

### 2) CLI 直接调用

```bash
pynlptool "南京市长江大桥"
pynlptool "南京市长江大桥" -o tags
pynlptool "南京市长江大桥" -o cut
pynlptool "南京市长江大桥" -o pos
pynlptool "南京市长江大桥" -o both
```

如果系统里没有 `pynlptool` 命令，也可以用：

```bash
python -m pynlptool.cli "南京市长江大桥" -o both
```

## Python API 快速示例

### 1) 使用内置模型（推荐）

```python
from pynlptool import load_model

# 显式加载模型后重复调用，适合批量推理场景
model = load_model()
print(model.cut("南京市长江大桥"))
# decode 输入为“观测序列”（这里是字符列表），输出为标签序列
print(model.decode(list("南京市长江大桥")))
```

### 2) 训练自定义模型

```python
from pynlptool import train

# 训练样本格式: (观测序列, 标签序列)
# 两个序列必须等长，且与 BMES 标注体系一致
sequences = [
    (["今", "天", "天", "气", "不", "错"], ["B", "E", "B", "E", "B", "E"]),
    (["我", "喜", "欢", "编", "程"], ["S", "B", "E", "B", "E"]),
]

# alpha: 拉普拉斯平滑系数，越大越平滑
# min_freq: 观测最小词频阈值，低频项会映射到 <UNK>
model = train(sequences, alpha=0.5, min_freq=1)
# 保存为可复用模型文件
model.save("your_model.pkl")
```

### 3) 训练时启用 BMM 观测增强

```python
from pynlptool import train

# 在不改变 HMM 结构的前提下，启用词典特征增强观测
# use_dict_feature=True: 开启 BMM 特征注入
# feature_joiner="|": 将“字|标签”拼接成新观测
# bmm_max_word_len=6: BMM 最大匹配词长
model = train(
    sequences,
    alpha=0.5,
    min_freq=1,
    use_dict_feature=True,
    feature_joiner="|",
    bmm_max_word_len=6,
)
```

### 4) 评估模型

```python
from pynlptool import load_data, evaluate, report

# 读取标注测试集（每行: 字 标签；句间空行）
test_sentences = load_data("test.txt")
# 转为 evaluate 所需结构
test_sequences = [(s.observations, s.tags) for s in test_sentences]
# 逐句推理，得到预测标签序列
predictions = [model.decode(obs) for obs, _ in test_sequences]

# 输出准确率、宏平均 P/R/F1 等指标
metrics = evaluate(test_sequences, predictions)
print(report(metrics))
```

### 5) 动态回退推理

```python
from pynlptool import infer_with_fallback, load_baseline_model, load_model

baseline = load_baseline_model()
bmm = load_model()

result = infer_with_fallback(baseline, bmm, "南京市长江大桥")
print(result.chosen_model)
print(" / ".join(result.words))
print(" | ".join(f"{word}/{pos}" for word, pos in result.words_pos))
```

## CLI

安装后可直接使用：

```bash
pynlptool "南京市长江大桥"
```

如果命令不存在（常见于 Windows PATH 未包含 Scripts）：

```bash
python -m pynlptool.cli "南京市长江大桥"
```

常用参数：

```bash
pynlptool "南京市长江大桥" -o cut
pynlptool "南京市长江大桥" -o tags
pynlptool "南京市长江大桥" -o pos
pynlptool "南京市长江大桥" -m your_model.pkl
```

- `-o, --output-format`: `tags` / `cut` / `pos` / `both`
- `-m, --model`: 指定模型路径
- 默认不指定 `-m` 时，CLI 会在内置基线模型与 BMM-HMM 之间执行动态回退
- `--disagreement-threshold`、`--min-avg-margin`、`--max-avg-score-drop`: 回退阈值
- `--version`: 查看版本

## 数据格式

训练文件采用“每行一个 字符 标签，句间空行分隔”：

```text
南 B_LOC
京 M_LOC
市 E_LOC
长 B_LOC
江 E_LOC
大 B_n
桥 E_n
```

## 核心 API

- `load_model()`: 加载内置 BMM 增强预训练模型（带缓存）
- `cut(text)`: 返回分词列表
- `tags(text)`: 返回与 CLI `-o tags` 一致的输出结果
- `pos(text)`: 返回与 CLI `-o pos` 一致的输出结果
- `both(text)`: 返回与 CLI `-o both` 一致的输出结果
- `train(...)`: 训练 HMM，支持 BMM 观测增强参数
- `evaluate(...)`: 计算准确率与宏平均指标

## 说明与边界行为

- `model.cut("")` 返回空列表
- 数字/英文会先归一化再解码，输出仍保留原字符
- 支持中英数混合输入

## 许可证

MIT License
