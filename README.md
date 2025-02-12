# ***评论&笔记***

- [Transformer (Google Brain, Google Research, 2017.6)](#Transformer)
- [BERT (Google AI Language, 2018.10)](#BERT)
- [ViT (Google Research, Brain, 2020.10)](#ViT)
- [MAE (Facebook AI Research (FAIR), 2021.11)](#MAE)
- [MoCo (Facebook AI Research (FAIR), 2019.11)](#MoCo)
- [Swin Transformer (Microsoft Research Asia, 2021.3)](#Swin Transformer)
- [CLIP (OpenAI, 2021.2)](#CLIP)
- [Codex (OpenAI, 2021.7)](#Codex)
- [AlphaCode (DeepMind, 2022.2)](#AlphaCode)
- [Scaling Laws (OpenAI, 2020.1)](#Scaling Laws)
- [T5 (Google, 2019.10)](#T5)
- [GPT 1 2 3 (OpenAI, 2018.6, 2019.2, 2020.5)](#GPT 1 2 3)
- [InstructGPT (OpenAI, 2022.3)](#InstructGPT)
- [Claude (Anthropic, 2022.4)](#Claude)
- [Llama 3 (Meta, 2024.7)](#Llama 3)
- [Mistral AI Models](#Mistral_AI_Models)
- [MoE (Google Brain, 2017.1)](#MoE)
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

# Attention Is All You Need
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

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
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

# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- Google Research, Brain, 2020.10

# Masked Autoencoders Are Scalable Vision Learners
- Facebook AI Research (FAIR), 2021.11

# Momentum Contrast for Unsupervised Visual Representation Learning
- Facebook AI Research (FAIR), 2019.11

# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- Microsoft Research Asia, 2021.3

# Learning Transferable Visual Models From Natural Language Supervision
- OpenAI, 2021.2

# Evaluating Large Language Models Trained on Code
- OpenAI, 2021.7

# Competition-Level Code Generation with AlphaCode
- DeepMind, 2022.2

# Scaling Laws for Neural Language Models
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

# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- Google, 2019.10

# GPT 1 2 3
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

# Training language models to follow instructions with human feedback
- OpenAI, 2022.3

<p align = "center">
<img src=/img/InstructGPT_method.png width="600" />
</p>

- 第二步和第三步实际是迭代进行的，可能会循环多次
- RM (reward model)和PPO policy第一次训练的参数初始化都是来自于SFT (supervised fine-tuning model)，但是PPO value func都是初始化自RM，之和的训练相当于就是持续学习了 (continual learning)，不过只有RM会用到过去收集的数据，PPO不会（这还是遵循on-policy RL的规律）
- 他只用了来自GPT 3的6B模型训练RM，他说175B不稳定
- RM Loss，其中 (k 2) 是组合符号，表示从k个样本中选取两两配对的数量，yw表示相比yl是更符合人类偏好的PPO policy生成的答案，r theta是RM，sigma是sigmoid函数

<div align="center">

$$ \mathcal{L}(\theta) = -\frac{1}{\binom{k}{2}} \sum_{(x, y_w, y_l) \sim D} \log [\sigma( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) )] $$

</div>

- PPO-ptx Object，其中r_theta是reward model，RL是PPO policy，SFT是supervised fine-tuning policy (model)，beta和gamma是设置的系数，第一项的(x, y)是采样自SFT，第二项的x是采样自预训练数据集，原文表达不严谨，x和y实际都是token序列，并不是一步输出的，是多步输出的序列，所以这里的概率都是条件概率，如下所示，所以项2相当于是最大化出现这一系列tokens（来自预训练数据）的概率，基本等价于原预训练loss，所以可以认为最大化这个Object，既起到了遵循RM指导的作用（并且不会过于偏离SFT），又起到了遵循预训练数据（分布）的作用

<div align="center">

$$ \mathcal{Object} = E_{(x, y) \sim D_{\text{RL}}}[r_{\theta} - \beta \log (RL(y | x) / SFT(y | x))] + \gamma E_{x \sim {\text{pretrain}}}[\log (RL(x))] $$
$$ RL(x) = \prod_{i=1}^{T} \pi(x_i | x_{<i}) $$
$$ \log(RL(x)) = \sum_{i=1}^{T} \log(\pi(x_i | x_{<i})) $$

</div>

- 这篇文章写的确实相当具有迷惑性，一开始我以为他是把PPO的clip surrogate object改成了以上的object直接优化，但是仔细翻看了多遍，发现这个object真是个十分具有诱导性的说法，他这个object里，第一项作为reward肯定是无异议的，但是根据[原RLHF论文](https://arxiv.org/abs/2009.01325)，他还是个多步强化学习问题，这个reward只在最后一步奖励（但是他的gamma设置等于1），其他步大概率就是0奖励了，所以他才非要有一个value func做信用分配，当然他单步也避免了动作太多，求条件概率时多概率相乘导致数值溢出，第二项肯定不能作为reward奖励，所以第二项应该是在update参数的时候作为单独的一项与PPO的loss结合（类似于entropy loss），再结合原RLHF论文和[Anthropic RLHF](https://arxiv.org/abs/2204.05862)来看，他第一项中r_theta后面那一项应该是求这两个分布的KL divergence，所以比较合理的做法以Anthropic RLHF的reward作为reward，再把他这个公式中的第二项放在ppo loss里面（当然肯定得取个负号）

# Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
- Anthropic, 2022.4
- 使用了[OOD detection techniques](https://arxiv.org/abs/2106.03004)来让模型回答：“不知道”，这个技术通常被称为ODIN(Out-of-Distribution detector for Neural networks)，它包含两个关键计算1.温度缩放(Temperature Scaling)2.输入扰动(Input Preprocessing)，整个过程简单来说就是让输入softmax之前的logits除以一个温度系数T（通常大于1），缩小概率分布的方差（差距），然后令被选出的类（概率最大的类）的对数概率对于输入x求导，令倒数乘以一个“步长”epsilon（一个较小的正数），然后用于改变x，最后再带入算softmax概率分布（依然要带入T），最大概率值小于阈值，则认为是OOD
- 文章提到他们有部署在线上并且持续循环训练RM和RL policy的步骤，结果显著改善了模型
- 用ELO score训练RM，没看懂这是什么
- reward奖励计算（个人认为原RLHF论文、InstructGPT和这篇论文的这个计算方式应该是一样的，所以InstructGPT的reward计算实现也应该参考这个）

<div align="center">

$$ reward = r_{PM} - \lambda_{KL} D_{KL}(policy || policy_0) $$

</div>

# The Llama 3 Herd of Models
- Meta, 2024.7
- 沐神说现在很多llm都是支持的8k上下文，训练的时候上下文是8k，但是部署的时候可以是32k，从实用上，32k的上下文长度对llm就够了，128k就更够了
- 沐神说文章没有给出具体的数据采样方法，就是在什么训练时期，哪些类型的数据（比如数学、code）的采样率是多少，这个数据采样率也十分重要

# Mistral_AI_Models
- Mistral AI
- 是原Llama 1团队出来创业的成果，[是一系列模型](https://docs.mistral.ai/getting-started/models/models_overview/)
- 听说Mistral Large 2比Llama3.1擅长代码和数学

# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- Google Brain, 2017.1

# Robust Speech Recognition via Large-Scale Weak Supervision
- OpenAI, 2022.12

# Noise2Music: Text-conditioned Music Generation with Diffusion Models
- Google Research, 2023.2
- [paper](https://arxiv.org/abs/2302.03917)，[blog](https://google-research.github.io/noise2music/)

# DALL-E 1 2 3
- OpenAI

# AlphaFold 1 2 3
- DeepMind

# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision
- Korea NAVER AI, 2021.2

# Align before Fuse: Vision and Language Representation Learning with Momentum Distillation
- Salesforce Research, 2021.7

# VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts
- Microsoft, 2021.11

# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
- Salesforce Research, 2022.1

# CoCa: Contrastive Captioners are Image-Text Foundation Models
- Google Research, 2022.5

# Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks
- Microsoft Corporation, 2022.8

# Movie Gen: A Cast of Media Foundation Models
- Meta, 2024.10

# HunyuanVideo: A Systematic Framework For Large Video Generative Models
- Tencent Hunyuan, 2024.12