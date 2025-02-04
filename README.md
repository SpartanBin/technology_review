# 评论&笔记

- [Transformer (Google Brain, Google Research, 2017.6)](#202502021731)
- [BERT (Google AI Language, 2018.10)](#202502021732)
- [ViT (Google Research, Brain, 2020.10)](#202502021733)
- [MAE (Facebook AI Research (FAIR), 2021.11)](#202502021734)
- [MoCo (Facebook AI Research (FAIR), 2019.11)](#202502021735)
- [Swin Transformer (Microsoft Research Asia, 2021.3)](#202502021736)
- [CLIP (OpenAI, 2021.2)](#202502021737)
- [Codex (OpenAI, 2021.7)](#202502021738)
- [AlphaCode (DeepMind, 2022.2)](#202502021739)
- [GPT 1 2 3 (OpenAI)](#202502021740)
- [InstructGPT (OpenAI, 2022.3)](#202502021741)
- [Claude (Anthropic, 2022.4)](#202502021742)
- [Llama 3 (Meta, 2024.7)](#202502021743)
- [Mistral AI](#202502022356)
- [Whisper (OpenAI, 2022.12)](#202502021744)
- [Noise2Music (Google Research, 2023.2)](#202502030008)
- [DALL-E 1 2 3 (OpenAI)](#202502021745)
- [AlphaFold 1 2 3 (DeepMind)](#202502021746)
- [ViLT (Korea NAVER AI, 2021.2)](#202502021747)
- [ALBEF (Salesforce Research, 2021.7)](#202502021748)
- [VLMo (Microsoft, 2021.11)](#202502021749)
- [BLIP (Salesforce Research, 2022.1)](#202502021750)
- [CoCa (Google Research, 2022.5)](#202502021751)
- [BEiT-3 (Microsoft Corporation, 2022.8)](#202502021752)
- [Movie Gen (Meta, 2024.10)](#202502021753)
- [HunyuanVideo (Tencent Hunyuan, 2024.12)](#202502021754)

## <span id="202502021731"> Attention Is All You Need </span>
- Google Brain, Google Research, 2017.6

<p align = "center">
<img src=/img/transformer_entirety.png width="400" />
</p>

- 多头注意力实现(pytorch)，参考的[MAPPO, 2021](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/mat/algorithm/ma_transformer.py)

<div align="center">

$$ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

</div>

```python
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, L, masked=False):

        super().__init__()

        assert d_model % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # output projection
        self.proj = nn.Linear(d_model, d_model)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)

    def forward(self, key, value, query):

        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y
```
- k、q、v都是同一个tensor经过linear transformation得到的（至少对于encoder是这样），对于decoder来说k、v的原tensor来自encoder的输出
- 如果堆叠多层decoder，每一层的k、v的原tensor都是同一个tensor，就是encoder的输出
- 激活函数使用的ReLU
- residual是先加后接LayerNorm
- Positional Encoding实现，GPT生成

<div align="center">

$$ PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right) $$
$$ PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right) $$

</div>

```python
import numpy as np

def positional_encoding(seq_length, d_model):
    """
    生成 Positional Encoding 矩阵。
    
    :param seq_length: 序列长度
    :param d_model: 词嵌入维度
    :return: 形状为 (seq_length, d_model) 的位置编码矩阵
    """
    position = np.arange(seq_length)[:, np.newaxis]  # (seq_length, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # shape = (d_model / 2,)
    
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度使用 sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度使用 cos
    
    return pe
```
- 这个位置编码与原tensor相加就应用了该位置编码
- div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))是在计算 10000^(2i/d_model) 的倒数，即 1 / (10000^(2i/d_model))，控制不同维度的位置编码缩放
- 未解决的疑问：transformer原本是被设计出来解决翻译问题的，翻译问题的encoder输入肯定是被翻译语言，decoder的输入是什么？还有就是encoder的输入的L和decoder的L不一样长怎么办，是否要求一样长？

## <span id="202502021732"> BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding </span>
- Google AI Language, 2018.10

<p align = "center">
<img src=/img/bert_pretrain.png width="300" />
<img src=/img/bert_embedding.png width="500" />
</p>

- BERT由transformer的encoder堆叠而成，所以是没有使用transformer原文中decoder使用的mask的
- BERT有三个可学习的embedding，就是word embedding、position embedding和segment embedding，最后将三者相加作为token的最终embedding，注意都是可学习的(比如用nn.Embedding实现)，这和transformer就不一样了
- word embedding就是语言模型都有的word embedding，position embedding用来表示位置（如transformer中的Positional Encoding），segment embedding基本上bert类模型独有，用来区分不同的句子
- bert的token中有三种特殊字符，分别是[CLS]、[SEP]和[MASK]，[CLS]的最终embedding用于下游分类任务，[SEP]放在一个句子的结尾（英文一个sequence可以最多由两个句子组成），[MASK]用于表示需要模型预测的词语（也就是预训练任务1拿去算loss的词语）
- 预训练任务有两种，1上面有提到，2是NSP，就是判断sequence中句子B是不是句子A的下一句，为了下游任务的鲁棒性，任务1模型输入中需要预测的词语有80%会被换成[MASK]，有10%会变成一个随机词语，剩下的直接不替换

## <span id="202502021733"> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale </span>
- Google Research, Brain, 2020.10

## <span id="202502021734"> Masked Autoencoders Are Scalable Vision Learners </span>
- Facebook AI Research (FAIR), 2021.11

## <span id="202502021735"> Momentum Contrast for Unsupervised Visual Representation Learning </span>
- Facebook AI Research (FAIR), 2019.11

## <span id="202502021736"> Swin Transformer: Hierarchical Vision Transformer using Shifted Windows </span>
- Microsoft Research Asia, 2021.3

## <span id="202502021737"> Learning Transferable Visual Models From Natural Language Supervision </span>
- OpenAI, 2021.2

## <span id="202502021738"> Evaluating Large Language Models Trained on Code </span>
- OpenAI, 2021.7

## <span id="202502021739"> Competition-Level Code Generation with AlphaCode </span>
- DeepMind, 2022.2

## <span id="202502021740"> GPT 1 2 3 </span>
- OpenAI

## <span id="202502021741"> Training language models to follow instructions with human feedback </span>
- OpenAI, 2022.3

## <span id="202502021742"> Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback </span>
- Anthropic, 2022.4

## <span id="202502021743"> The Llama 3 Herd of Models </span>
- Meta, 2024.7
- 沐神说现在很多llm都是支持的8k上下文，训练的时候上下文是8k，但是部署的时候可以是32k，从实用上，32k的上下文长度对llm就够了，128k就更够了
- 沐神说文章没有给出具体的数据采样方法，就是在什么训练时期，哪些类型的数据（比如数学、code）的采样率是多少，这个数据采样率也十分重要

## <span id="202502022356"> Mistral AI Models </span>
- Mistral AI
- 是原Llama 1团队出来创业的成果，[是一系列模型](https://docs.mistral.ai/getting-started/models/models_overview/)
- 听说Mistral Large 2比Llama3.1擅长代码和数学

## <span id="202502021744"> Robust Speech Recognition via Large-Scale Weak Supervision </span>
- OpenAI, 2022.12

## <span id="202502030008"> Noise2Music: Text-conditioned Music Generation with Diffusion Models </span>
- Google Research, 2023.2
- [paper](https://arxiv.org/abs/2302.03917)，[blog](https://google-research.github.io/noise2music/)

## <span id="202502021745"> DALL-E 1 2 3 </span>
- OpenAI

## <span id="202502021746"> AlphaFold 1 2 3 </span>
- DeepMind

## <span id="202502021747"> ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision </span>
- Korea NAVER AI, 2021.2

## <span id="202502021748"> Align before Fuse: Vision and Language Representation Learning with Momentum Distillation </span>
- Salesforce Research, 2021.7

## <span id="202502021749"> VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts </span>
- Microsoft, 2021.11

## <span id="202502021750"> BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation </span>
- Salesforce Research, 2022.1

## <span id="202502021751"> CoCa: Contrastive Captioners are Image-Text Foundation Models </span>
- Google Research, 2022.5

## <span id="202502021752"> Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks </span>
- Microsoft Corporation, 2022.8

## <span id="202502021753"> Movie Gen: A Cast of Media Foundation Models </span>
- Meta, 2024.10

## <span id="202502021754"> HunyuanVideo: A Systematic Framework For Large Video Generative Models </span>
- Tencent Hunyuan, 2024.12