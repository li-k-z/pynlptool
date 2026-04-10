"""
HMM 模型实现及 Viterbi 解码
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pynlptool.data_utils import augment_observations_with_bmm, build_dictionary_from_sequences, norm_seq


@dataclass
class HMM:
    """
    隐马尔可夫模型（HMM）用于序列标注。

    Attributes:
        states: 所有可能的隐状态（标签）列表
        vocab: 词表中所有观察符号列表
        log_pi: 初始状态对数概率
        log_a: 状态转移对数概率
        log_b: 发射对数概率
        log_unk: 每个状态的未知词对数概率
        unk_token: 未知词标记
        tag_dict: 词到可能标签的映射字典
        tag_penalty: 已知词使用词典外标签的惩罚
        log_end: 每个状态的结束对数概率
    """

    states: List[str]
    vocab: List[str]
    log_pi: Dict[str, float]
    log_a: Dict[str, Dict[str, float]]
    log_b: Dict[str, Dict[str, float]]
    log_unk: Dict[str, float]
    unk_token: str = "<UNK>"
    tag_dict: Optional[Dict[str, Set[str]]] = None
    tag_penalty: float = -20.0
    log_end: Optional[Dict[str, float]] = None
    use_dict_feature: bool = False
    dict_lexicon: Optional[Set[str]] = None
    feature_joiner: str = "|"
    bmm_max_word_len: int = 6

    def save(self, path: str) -> None:
        """保存模型到文件。"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HMM":
        """从文件加载模型。"""
        import io
        
        class _ModuleRemapper(pickle.Unpickler):
            """自定义Unpickler，将旧模块名映射到新模块名。"""
            def find_class(self, module: str, name: str):
                # 将旧的 nlptool 模块名映射到 pynlptool
                if module.startswith("nlptool"):
                    module = module.replace("nlptool", "pynlptool", 1)
                return super().find_class(module, name)
        
        with open(path, "rb") as f:
            model: HMM = _ModuleRemapper(f).load()
        # Backward compatibility: older pickles may miss new fields
        if model.tag_dict is None:
            model.tag_dict = {}
        if not hasattr(model, "tag_penalty"):
            model.tag_penalty = -20.0
        if not hasattr(model, "log_end") or model.log_end is None:
            uniform = math.log(1.0 / max(len(model.states), 1)) if model.states else 0.0
            model.log_end = {s: uniform for s in model.states}
        if not hasattr(model, "use_dict_feature"):
            model.use_dict_feature = False
        if not hasattr(model, "dict_lexicon") or model.dict_lexicon is None:
            model.dict_lexicon = set()
        if not hasattr(model, "feature_joiner"):
            model.feature_joiner = "|"
        if not hasattr(model, "bmm_max_word_len"):
            model.bmm_max_word_len = 6
        return model

    def _is_augmented_observation(self, obs: str) -> bool:
        if self.feature_joiner not in obs:
            return False
        left, _, right = obs.rpartition(self.feature_joiner)
        return bool(left) and right in {"B", "M", "E", "S"}

    def _prepare_observations(self, observations: List[str]) -> List[str]:
        """按配置将原始观测转换为训练时一致的观测表示。"""
        if not self.use_dict_feature:
            return observations
        if all(self._is_augmented_observation(o) for o in observations):
            return observations
        lexicon = self.dict_lexicon or set()
        return augment_observations_with_bmm(
            observations,
            lexicon=lexicon,
            joiner=self.feature_joiner,
            max_word_len=self.bmm_max_word_len,
        )

    def decode(self, observations: List[str]) -> List[str]:
        """
        Viterbi 解码，找到最可能的状态序列。

        Args:
            observations: 观察符号列表（如字符列表）

        Returns:
            预测的标签序列
        """
        if not observations:
            return []

        observations = self._prepare_observations(observations)

        states = self.states
        vocab_set = set(self.vocab)
        tag_dict = self.tag_dict or {}
        T = len(observations)

        dp: List[Dict[str, float]] = []
        bp: List[Dict[str, str]] = []

        def split_bmes(tag: str) -> Tuple[Optional[str], Optional[str]]:
            if not tag:
                return None, None
            p = tag[0]
            if p not in {"B", "M", "E", "S"}:
                return None, None
            if len(tag) == 1:
                return p, ""
            if tag[1] in {"_", "-", "/"}:
                return p, tag[2:]
            return p, tag[1:] 

        def valid_transition(prev_tag: str, curr_tag: str) -> bool:
            pp, pt = split_bmes(prev_tag)
            cp, ct = split_bmes(curr_tag)
            if cp is None or pp is None:
                return True
            if pp in {"E", "S"}:
                return cp in {"B", "S"}
            if pp in {"B", "M"}:
                if cp not in {"M", "E"}:
                    return False
                return pt == ct
            return True

        def valid_start(curr_tag: str) -> bool:
            cp, _ = split_bmes(curr_tag)
            if cp is None:
                return True
            return cp in {"B", "S"}

        def valid_end(last_tag: str) -> bool:
            lp, _ = split_bmes(last_tag)
            if lp is None:
                return True
            return lp in {"E", "S"}

        first_obs = observations[0] if observations[0] in vocab_set else self.unk_token
        dp0: Dict[str, float] = {}
        bp0: Dict[str, str] = {}
        allowed0: Set[str] = tag_dict.get(first_obs, set(states)) if first_obs in vocab_set else set(states)
        for s in states:
            emit = self.log_b[s].get(first_obs, self.log_unk[s])
            penalty = self.tag_penalty if (first_obs in vocab_set and s not in allowed0) else 0.0
            dp0[s] = (self.log_pi[s] + emit + penalty) if valid_start(s) else -math.inf
            bp0[s] = ""
        dp.append(dp0)
        bp.append(bp0)

        for t in range(1, T):
            obs = observations[t] if observations[t] in vocab_set else self.unk_token
            allowed_t: Set[str] = tag_dict.get(obs, set(states)) if obs in vocab_set else set(states)
            dp_t: Dict[str, float] = {}
            bp_t: Dict[str, str] = {}
            for s in states:
                emit = self.log_b[s].get(obs, self.log_unk[s])
                penalty = self.tag_penalty if (obs in vocab_set and s not in allowed_t) else 0.0
                best_prev = None
                best_score = -math.inf
                for s_prev in states:
                    if not valid_transition(s_prev, s):
                        continue
                    score = dp[t - 1][s_prev] + self.log_a[s_prev][s] + emit + penalty
                    if score > best_score:
                        best_score = score
                        best_prev = s_prev
                dp_t[s] = best_score
                bp_t[s] = best_prev or states[0]
            dp.append(dp_t)
            bp.append(bp_t)

        end_scores: Dict[str, float] = {}
        for s in states:
            end_scores[s] = (dp[-1][s] + self.log_end.get(s, 0.0)) if valid_end(s) else -math.inf
        last_state = max(end_scores, key=lambda k: end_scores[k])
        path = [last_state]
        for t in range(T - 1, 0, -1):
            last_state = bp[t][last_state]
            path.append(last_state)
        path.reverse()
        return path
    
    def cut(self, text: str) -> List[str]:
        """
        中文分词。

        Args:
            text: 输入的中文文本

        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        chars = list(text)
        normalized = norm_seq(chars)
        tags = self.decode(normalized)
        
        words = []
        current_word = ""
        for char, tag in zip(chars, tags):
            pos = tag[0] if tag else "S"
            if pos in {"B", "S"}:
                if current_word:
                    words.append(current_word)
                current_word = char
            else:  # M or E
                current_word += char
        if current_word:
            words.append(current_word)
        
        return words


def train(
    sequences: Iterable[Tuple[List[str], List[str]]],
    alpha: float = 0.1,
    min_freq: int = 3,
    unk_token: str = "<UNK>",
    tag_penalty: float = -20.0,
    use_dict_feature: bool = False,
    dict_lexicon: Optional[Set[str]] = None,
    feature_joiner: str = "|",
    bmm_max_word_len: int = 6,
) -> HMM:
    """
    训练 HMM 模型。

    Args:
        sequences: (observations, tags) 元组的可迭代对象
        alpha: 平滑参数
        min_freq: 词频阈值，低于此值视为未知词
        unk_token: 未知词标记
        tag_penalty: 标签惩罚系数
        use_dict_feature: 是否启用 BMM 词典观测增强
        dict_lexicon: 可选外部词典（不传则由训练集自动构建）
        feature_joiner: 观测拼接符号
        bmm_max_word_len: BMM 最大词长

    Returns:
        训练好的 HMM 模型
    """
    start_counts: Counter = Counter()
    end_counts: Counter = Counter()
    trans_counts: Dict[str, Counter] = defaultdict(Counter)
    emit_counts: Dict[str, Counter] = defaultdict(Counter)
    state_counts: Counter = Counter()
    vocab: Counter = Counter()

    sequences_list: List[Tuple[List[str], List[str]]] = []

    num_sentences = 0
    for obs, tags in sequences:
        if not obs or not tags:
            continue
        sequences_list.append((obs, tags))
        num_sentences += 1
        start_counts[tags[0]] += 1
        state_counts.update(tags)
        vocab.update(obs)

    if use_dict_feature:
        lexicon = set(dict_lexicon) if dict_lexicon is not None else build_dictionary_from_sequences(sequences_list)
        transformed: List[Tuple[List[str], List[str]]] = []
        for obs, tags in sequences_list:
            transformed_obs = augment_observations_with_bmm(
                obs,
                lexicon=lexicon,
                joiner=feature_joiner,
                max_word_len=bmm_max_word_len,
            )
            transformed.append((transformed_obs, tags))
        sequences_list = transformed
    else:
        lexicon = set()

    vocab = Counter()
    for obs, _ in sequences_list:
        vocab.update(obs)

    vocab_set = {w for w, c in vocab.items() if c >= min_freq}
    vocab_set.add(unk_token)

    for obs, tags in sequences_list:
        for i in range(len(tags)):
            token = obs[i] if obs[i] in vocab_set else unk_token
            emit_counts[tags[i]][token] += 1
            if i == len(tags) - 1:
                end_counts[tags[i]] += 1
            if i > 0:
                trans_counts[tags[i - 1]][tags[i]] += 1

    states = sorted(state_counts.keys())
    vocab_list = sorted(vocab_set)
    S = len(states)
    V = len(vocab_list)

    log_pi: Dict[str, float] = {}
    log_a: Dict[str, Dict[str, float]] = {}
    log_b: Dict[str, Dict[str, float]] = {}
    log_unk: Dict[str, float] = {}
    log_end: Dict[str, float] = {}

    for s in states:
        log_pi[s] = math.log((start_counts[s] + alpha) / (num_sentences + alpha * S))

    for s in states:
        log_a[s] = {}
        trans_total = sum(trans_counts[s].values())
        denom = trans_total + alpha * S
        for s2 in states:
            log_a[s][s2] = math.log((trans_counts[s][s2] + alpha) / denom)

    tag_dict: Dict[str, Set[str]] = defaultdict(set)
    for s in states:
        log_b[s] = {}
        emit_total = sum(emit_counts[s].values())
        denom = emit_total + alpha * V
        for w, c in emit_counts[s].items():
            log_b[s][w] = math.log((c + alpha) / denom)
            if w != unk_token:
                tag_dict[w].add(s)
        log_unk[s] = math.log(alpha / denom)

    for s in states:
        denom = end_counts[s] + alpha * 1
        log_end[s] = math.log((end_counts[s] + alpha) / denom)

    return HMM(
        states=states,
        vocab=vocab_list,
        log_pi=log_pi,
        log_a=log_a,
        log_b=log_b,
        log_unk=log_unk,
        unk_token=unk_token,
        tag_dict=dict(tag_dict),
        tag_penalty=tag_penalty,
        log_end=log_end,
        use_dict_feature=use_dict_feature,
        dict_lexicon=lexicon,
        feature_joiner=feature_joiner,
        bmm_max_word_len=bmm_max_word_len,
    )
