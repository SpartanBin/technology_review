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
- [Scaling Laws (OpenAI, 2020.1)](#202502091904)
- [T5 (Google, 2019.10)](#202502092120)
- [GPT 1 2 3 (OpenAI, 2018.6, 2019.2, 2020.5)](#202502021740)
- [InstructGPT (OpenAI, 2022.3)](#202502021741)
- [Claude (Anthropic, 2022.4)](#202502021742)
- [Llama 3 (Meta, 2024.7)](#202502021743)
- [Mistral AI](#202502022356)
- [MoE (Google Brain, 2017.1)](#202502050303)
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
- Feed Forward只作用于最后一个维度(d_model)，就是一个两层FCN，宽度通常大于d_model，先把d_model拉大，再还原到原大小，激活函数使用的ReLU
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

## <span id="202502091904"> Scaling Laws for Neural Language Models </span>
- OpenAI, 2020.1
- FLOPs即浮点运算次数，Transformer模型的前向传播中每个参数大约需要2次浮点运算，反向传播大约需要4次，从而每个参数每处理一个token总共约6次FLOPs（训练时），因此，训练过程中总计算量可以近似表示为，C = 6 * N * D，N是模型参数量，D是过模型的token数量
- 当数据充分计算量充分的时候，训练(validation loss)损失与模型大小的幂函数的倒数间基本呈正比关系（线性关系）

<div align="center">

$$ L(N) \propto N^{-\alpha} $$

</div>

- 训练损失与数据量同上

<div align="center">

$$ L(D) \propto D^{-\beta} $$

</div>

- 计算量对损失的影响

<div align="center">

$$ L(C) \propto C^{-\gamma} $$

</div>

- 简而言之，训练模型结果是可以预测的（利用结构相同但是规模较小的模型结果），计算量上升模型表现也会上升，但是边际效应递减，并不能无限增大
- 当预算有限的情况下，增大模型参数数量比增大数据量更有效
- 小模型相比大模型更容易过拟合，大模型泛化的潜力更强

## <span id="202502092120"> Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer </span>
- Google, 2019.10

## <span id="202502021740"> GPT 1 2 3 </span>
- OpenAI, 2018.6, 2019.2, 2020.5
- (GPT 1) Improving Language Understanding by Generative Pre-Training
- (GPT 2) Language Models are Unsupervised Multitask Learners
- (GPT 3) Language Models are Few-Shot Learners
- GPT 1号称使用[multi-layer Transformer decoder](https://arxiv.org/abs/1801.10198)的结构，实际没有，以下才是它的结构

<p align = "center">
<img src=/img/gpt1_entirety.png width="300" />
</p>

- 所以和原transformer decoder相比，GPT 1的每个block少了一个接收来自encoder K、V的multi-head attention
- GPT 1使用可学习的position embedding
- 下游任务都是使用最后一个token的embedding进入不同的输出头
- 基本所有的类GPT 1自回归生成LM每次都只预测一个token
- 所有的类GPT 1自回归生成LM都是在推理(inference)时递归地生成内容，即在每一步预测时，模型输入的是之前所有生成的 token（加上可能的初始上下文），从而不断扩展文本，也就是说，推理时模型需要考虑所有过去生成的内容。但是在训练期间都使用teacher forcing，即使用真实的、已知的token序列作为输入
- 所以为了节省资源，类GPT 1自回归生成LM可以缓存先前的K、V cache，每次只计算新的一个token的Q、K、V，并继续储存K、V
- GPT 2指出语言模型的很多下游任务实际都可以用一种无监督的多任务学习方式学出来（这或许就是prompt的雏形），就是比如你想它做翻译，你就说：“从中文翻译成英文: 你是狗 =>”
- GPT 2模型结构基本和GPT 1一样（当然肯定更大了），除了改了layer norm的位置到每个block前，并且在最后一个block之后加了一个layer norm，以及在初始化后，将残差层的权重除以n^0.5，n是残差层的层数
- GPT 3提出了一种特殊的zero-shot、few-shot方式，他称其为in-context learning，就是在'prompt'里添加示例，不加示例就是zero，加多少个示例就是多少shot，比如以下one-shot例子：“从中文翻译成英文: 你是狗 => you are a dog 他是猪 =>”
- GPT 3结构与GPT 2相同，除了交替使用稀疏和稠密的attention，像[Sparse Transformer](https://arxiv.org/abs/1904.10509)那样，Sparse Transformer主要包含一个稀疏注意力模式和另一个块稀疏矩阵来储存，稀疏注意力模式有三类，一是跨步注意，就是只注意固定间隔token，二是固定模式，就是只关注自己附近的，就像卷积核那样，这两种在Sparse Transformer中会交替使用
- GPT 3提到在生成时使用了[beam search](https://en.wikipedia.org/wiki/Beam_search)，且和T5的设置一样，传统的beam search包含三个参数b、v、t，b代表候选数，v代表词汇表大小，t代表长度，传统beam search的计算复杂度为b * v * t，在LM里面，beam search还会有长度惩罚α，原因是因为LM生成停止符后会终止生成，这样句子会长度不一，短句的对数概率之和更容易大于长句，不利于长句的生成

<div align="center">

$$ S(Y) = \sum_{i=1}^{T} \log P(y_i | y_1,...,y_{i-1},x) $$
$$ S'(Y) = S(Y) \div (T^\alpha) $$

</div>

## <span id="202502021741"> Training language models to follow instructions with human feedback </span>
- OpenAI, 2022.3

<p align = "center">
<img src=/img/InstructGPT_method.png width="600" />
</p>

- 第二步和第三步实际是迭代进行的，可能会循环多次
- RM (reward model)和PPO (PPO policy)第一次训练的参数初始化都是来自于SFT (supervised fine-tuning model)，之和的训练相当于就是持续学习了 (continual learning)，不过只有RM会用到过去收集的数据，PPO不会（这还是遵循on-policy RL的规律）
- 他只用了来自GPT 3的6B模型训练RM，他说175B不稳定
- RM Loss，其中 (k 2) 是组合符号，表示从k个样本中选取两两配对的数量

<div align="center">

$$ \mathcal{L}(\theta) = -\frac{1}{\binom{k}{2}} \sum_{(y_w, y_l, x) \sim D} \log \[\sigma\( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) \)\] $$

</div>

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

## <span id="202502050303"> Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer </span>
- Google Brain, 2017.1

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