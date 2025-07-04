***注意！因gitee和github的markdown公式代码不完全一样，本文写作时是以gitee为准的，所以github版会有公式无法渲染或渲染错误的情况，[gitee地址](https://gitee.com/spartanbin/technology_review)，[github地址](https://github.com/SpartanBin/technology_review)***

# 大模型

- [Transformer (Google Brain, Google Research, 2017.6)](#202502021731)
- [BERT (Google AI Language, 2018.10)](#202502021732)
- [ViT (Google Research, Brain, 2020.10)](#202502021733)
- [MAE (Facebook AI Research (FAIR), 2021.11)](#202502021734)
- [MoCo (Facebook AI Research (FAIR), 2019.11)](#202502021735)
- [Swin Transformer (Microsoft Research Asia, 2021.3)](#202502021736)
- [CLIP (OpenAI, 2021.2)](#202502021737)
- [Codex (OpenAI, 2021.7)](#202502021738)
- [AlphaCode (DeepMind, 2022.2)](#202502021739)
- [bfloat16 (Google Brain)](#202503021144)
- [MoE (Google Brain, 2017.1)](#202502050303)
- [Scaling Laws (OpenAI, 2020.1)](#202502091904)
- [LoRA (Microsoft, 2021.6)](#202503021224)
- [Chain-of-Thought (Google Research, Brain, 2022.1)](#202502191230)
- [Toolformer (Meta AI Research, 2023.2)](#202502151904)
- [T5 (Google, 2019.10)](#202502092120)
- [GPT 1 2 3 (OpenAI, 2018.6, 2019.2, 2020.5)](#202502021740)
- [InstructGPT (OpenAI, 2022.3)](#202502021741)
- [Claude (Anthropic, 2022.4)](#202502021742)
- [DPO (Stanford University, 2023.5)](#202502191442)
- [Llama 1 2 3 (Meta, 2023.2, 2023.7, 2024.7)](#202502021743)
- [Mistral AI Models (Mistral AI, 2023.10, 2024.1, 2024.10)](#202502022356)
- [DeepSeek Models (DeepSeek-AI, 2024.1, 2024.1, 2024.2, 2024.3, 2024.5, 2024.12, 2025.1)](#202503302359)
- [Flamingo (DeepMind, 2022.4)](#202502151055)
- [Whisper (OpenAI, 2022.12)](#202502021744)
- [Noise2Music (Google Research, 2023.2)](#202502030008)
- [AlphaFold 1 2 3 (DeepMind)](#202502021746)
- [ViLT (Korea NAVER AI, 2021.2)](#202502021747)
- [ALBEF (Salesforce Research, 2021.7)](#202502021748)
- [VLMo (Microsoft, 2021.11)](#202502021749)
- [BLIP (Salesforce Research, 2022.1)](#202502021750)
- [CoCa (Google Research, 2022.5)](#202502021751)
- [BEiT-3 (Microsoft Corporation, 2022.8)](#202502021752)
- [DALL-E 1 2 3 (OpenAI, 2021.2, 2022.4, 2023.8)](#202502021745)
- [U-Net (University of Freiburg, Germany, 2015.5)](#202503051357)
- [VQ-VAE (DeepMind, 2017.11)](#202503131226)
- [VQ-GAN (CompVis, 2020.12)](#202503071100)
- [Stable Diffusion 1, SDXL, SD 3 (CompVis, Runway ML, Stability AI, 2021.12, 2023.7, 2024.3)](#202503051346)
- [Movie Gen (Meta, 2024.10)](#202502021753)
- [HunyuanVideo (Tencent Hunyuan, 2024.12)](#202502021754)
- [Stanford Town (Stanford University, 2023.4)](#202503021548)
- [QAT](#202503021530)

# 强化学习

- [Knowledge Review](#202506020019)
- [DQN (DeepMind, 2013.12)](#202505262258)
- [DDPG (DeepMind, 2015.9)](#202505262300)
- [Double DQN (DeepMind, 2015.9)](#202505272217)
- [TD3 (McGill University, 2018.2)](#202505262304)
- [soft Q-learning (UC Berkeley, 2017.2)](#202506072104)
- [SAC (UC Berkeley, 2018.1, 2018.12)](#202505262308)
- [C51 (DeepMind, 2017.7)](#202506072111)
- [QR-DQN (DeepMind, 2017.10)](#202506072114)
- [D4PG (DeepMind, 2018.4)](#202506072135)
- [IQN (DeepMind, 2018.6)](#202506072125)
- [Distributional-SAC (Tsinghua University, 2020.1, 2023.10)](#202506071230)
- [Diffusion-QL (Twitter, 2022.8)](#202506072143)
- [Decision Diffuser (MIT, 2022.11)](#202506072148)
- [Diffusion Policy (Columbia University, 2023.3)](#202506072151)
- [DACER (Tsinghua University, 2024.5, 2025.5)](#202506032314)

## <span id="202502021731"> Transformer </span>
- Google Brain, Google Research, 2017.6
- Attention Is All You Need

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

        kv_L = key.size()[1]
        B, q_L, D = query.size()

        # calculate key, value, query for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, kv_L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, kv_L, hs)
        v = self.value(value).view(B, kv_L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, kv_L, hs)
        q = self.query(query).view(B, q_L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, q_L, hs)

        # causal attention: (B, nh, q_L, hs) x (B, nh, hs, kv_L) -> (B, nh, q_L, kv_L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, q_L, kv_L) x (B, nh, kv_L, hs) -> (B, nh, q_L, hs)
        y = y.transpose(1, 2).contiguous().view(B, q_L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y
```
- k、q、v都是同一个tensor经过linear transformation得到的（至少对于encoder是这样），对于decoder来说，第一层多头的k、v来自decoder自己，第二层多头的k、v来自encoder，如果堆叠多个encoder、decoder块，decoder每一块的来自encoder的k、v都是一样的，就是encoder的最后一层输出，注意正是因为此特性，所以encoder和decoder的时间维度大小可以不一样（输入tokens序列长度可以不一样）
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
- transformer是被设计出来解决翻译问题的，因此encoder的输入是被翻译语言，decoder的输入是目标翻译语言，和LLM一样，decoder是自回归的，也就是一个一个token去预测的，因此inference的时候decoder的输入就是他刚才自己预测并输出的token，training的时候理应用teaching force就是强制和label一样

## <span id="202502021732"> BERT </span>
- Google AI Language, 2018.10
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

<p align = "center">
<img src=/img/bert_pretrain.png width="300" />
<img src=/img/bert_embedding.png width="500" />
</p>

- BERT由transformer的encoder堆叠而成，所以是没有使用transformer原文中decoder使用的mask的
- BERT有三个可学习的embedding，就是word embedding、position embedding和segment embedding，最后将三者相加作为token的最终embedding，注意都是可学习的(比如用nn.Embedding实现)，这和transformer就不一样了
- word embedding就是语言模型都有的word embedding，position embedding用来表示位置（如transformer中的Positional Encoding），segment embedding基本上bert类模型独有，用来区分不同的句子
- bert的token中有三种特殊字符，分别是[CLS]、[SEP]和[MASK]，[CLS]的最终embedding用于下游分类任务，[SEP]放在一个句子的结尾（英文一个sequence可以最多由两个句子组成），[MASK]用于表示需要模型预测的词语（也就是预训练任务1拿去算loss的词语）
- 预训练任务有两种，1上面有提到，2是NSP，就是判断sequence中句子B是不是句子A的下一句，为了下游任务的鲁棒性，任务1模型输入中需要预测的词语有80%会被换成[MASK]，有10%会变成一个随机词语，剩下的直接不替换

## <span id="202502021733"> ViT </span>
- Google Research, Brain, 2020.10
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

<p align = "center">
<img src=/img/vit_backbone.png width="800" />
</p>

- 上图表达的十分清楚，就是将图像切成若干16 * 16大小的块，然后按顺序组成tokens序列，使用的位置编码是1D位置编码，是从线性变换之后插入的，使用的应该是原PE而不是bert那种可学习的（也尝试了可学习的和其他多种包括2D的，差别不大），其他设置几乎和bert相同，比如使用了类似的可学习的[CLS]用作输出编码，使用了标准的transformer encoder作为block

## <span id="202502021734"> MAE </span>
- Facebook AI Research (FAIR), 2021.11
- Masked Autoencoders Are Scalable Vision Learners

<p align = "center">
<img src=/img/mae_training.png width="800" />
</p>

- 是一种图像预训练方法，是标准的auto-encoder结构，预训练完decoder就不要了
- mask是随机mask的，上图表达的十分清楚，mask的是像ViT那样的一块一块的，encoder是ViT，但是只接收没有mask的块，decoder就全部接收，decoder也是ViT，大小只有encoder的1/10
- loss是MSE，且只计算mask位置的

## <span id="202502021735"> MoCo </span>
- Facebook AI Research (FAIR), 2019.11
- Momentum Contrast for Unsupervised Visual Representation Learning

## <span id="202502021736"> Swin Transformer </span>
- Microsoft Research Asia, 2021.3
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## <span id="202502021737"> CLIP </span>
- OpenAI, 2021.2
- Learning Transferable Visual Models From Natural Language Supervision

<p align = "center">
<img src=/img/clip_training.png width="1000" />
</p>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, initial_temperature, img_emb, text_emb, emb):
        """
        初始化对比学习损失函数。
        参数：
        - initial_temperature: 初始的 temperature 值
        """
        super(ContrastiveLoss, self).__init__()
        # 定义 temperature 为可学习参数
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.img_proj = nn.Linear(img_emb, emb, bias=False)
        self.text_proj = nn.Linear(text_emb, emb, bias=False)

    def forward(self, image_features, text_features):
        """
        计算对比学习损失。
        参数：
        - image_features: 图像特征张量，形状为 [batch_size, img_emb]
        - text_features: 文本特征张量，形状为 [batch_size, text_emb]

        返回：
        - loss: 对比学习损失标量
        """
        # 确保特征已归一化
        I_e = F.normalize(self.img_proj(image_features), p=2, dim=1)  # (batch_size, emb)
        T_e = F.normalize(self.text_proj(text_features), p=2, dim=1)  # (batch_size, emb)

        # 计算相似度矩阵
        logits_for_img = (I_e @ T_e.T) / torch.exp(self.temperature)  # (batch_size, batch_size)
        logits_for_text = logits_for_img.T

        # 构造标签，在CLIP里image_features和text_features配对的样本是一一对应的
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size)

        # 计算交叉熵损失
        loss_i = F.cross_entropy(logits_for_img, labels)
        loss_t = F.cross_entropy(logits_for_text, labels)

        # 返回平均损失
        return (loss_i + loss_t) / 2
```

- CLIP使用的是对比学习方法，text encoder是GPT 2的架构，image encoder是ViT或者是ResNet的架构，训练使用的InfoNCE loss，因数据量够大，没有用预训练参数初始化两个encoder，是直接训练的，每次训练，都需要输入batch size对图像文本对，成对的就是正样本，其他batch size - 1个都是负样本，通过这种表示学习（对比学习）方法，相当于是把成对的向量距离拉近，不成对拉远
- 右图表示如何inference，比如选择要做分类任务，有1000类，那就用一个固定的语句格式把这1000类变成1000个语句，带入text enc，得到1000个向量，然后图片过img enc得到一个向量，然后算距离，距离最近的类就是分类结果，CLIP的成就就是它解放了类别标签，由它训练出的分类器并不局限于固定数量的类别限制了

## <span id="202502021738"> Codex </span>
- OpenAI, 2021.7
- Evaluating Large Language Models Trained on Code
- 这项技术最终变成了Github Copilot，论文里面只用了github python数据，但是Copilot可不止会写python
- 微调自GPT 12B的模型
- 高效实现了pass@k这种评价机制（详见原文），这种评价机制差不多就等于在给定多个候选解时，有多少概率至少有一个能通过所有测试

## <span id="202502021739"> AlphaCode </span>
- DeepMind, 2022.2
- Competition-Level Code Generation with AlphaCode
- 用了多个主流编程语言的所有github公开库数据，预训练是在github数据上做的，还用了编程竞赛数据微调
- 使用n@k评价机制，和Codex的pass@k是一样的
- 将编程理解成sequence-to-sequence translation task，因此使用标准的完整transformer结构，但是使用了[multi-query attention](https://arxiv.org/abs/1911.02150)，就是q保持不变，所有k、v都共享参数，可见grouped-query attention (GQA)就是传统和这种方法的折衷方案
- 预训练中有两个loss，用了bert的masked language modeling loss针对encoder，standard cross-entropy next-token prediction loss针对decoder，针对github文件随机采样一个位置，位置之前的作为encoder输入，之后的给decoder
- 微调阶段和预训练loss基本一样，但是将decoder的loss改成了修改后的GOLD（详见原文），因为竞赛数据里包含多种解法，修改后的GOLD可以让模型只关注一种解法（模型已经拟合了的解法），不然模型会增加每一种解法的概率（可能反而会影响正确率？），只是这次是把竞赛的问题描述给encoder，代码给decoder，微调时也用了竞赛里面错误的问题提交，用了两种针对此的解决方案，第一种是在问题描述中加入此问题是否正确的描述，另一种是准备了一个小的transformer接收来自主模型的最后一层token表示去判断是正确还是错误（sampling阶段不用）
- 在采样回答时也有多种特殊方法和处理（详见原文），比如使用了[nucleus sampling](https://arxiv.org/abs/1904.09751)（核采样，top-p采样），核采样不需要像beam search一样维持多个候选序列，核采样只有一个序列，采样方法是设置一个阈值p，按概率大小排列候选词，然后从大到小依次加这些概率，直到求和大小大于等于p截至，然后根据参与求和的这些词的概率（归一化后）采样出一个词，然后继续以上步骤，直到生成完整序列

## <span id="202503021144"> bfloat16 </span>
- Google Brain, brain floating point, [wiki](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)

<p align = "center">
<img src=/img/bfloat16_1.png width="500" />
<img src=/img/bfloat16_2.png width="800" />
</p>

- 现在训练模型基本都是用的bfloat16数据格式，torch.dtype是torch.bfloat16
- 可以看到bfloat16实际就是改了原本的float16 指数和尾数的位宽，bfloat16的指数位宽和float32是一样的，那它的数值范围和float32也一样了，但是牺牲了精度

## <span id="202502050303"> MoE </span>
- Google Brain, 2017.1
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

<p align = "center">
<img src=/img/moe_layer.png width="400" />
</p>

- 每一个MoE层由一个或两个(two-level hierarchical MoE)门控单元和多个专家n组成，门控单元会经过softmax输出n个概率，然后启用top k个专家，把他们的输出结果求和作为输出，在这篇论文中他们的专家用的LSTM，门控就是线性变换
- 原文提到他们是用类似dropout的机制实现的只让k个专家参与运算，可以想象相当于是不参与的就都不进行forward，然后他们会让gradient backward回来直接训练门控单元，而不是像[这篇论文](https://arxiv.org/abs/1511.06297)一样使用强化学习(Reinforce)训练
- 然后他们为了不让门控单元只去依赖某几个专家，他们用了Noisy Top-K Gating，就是增加了一个可学习的噪声参数，并让噪声与原门口输出相加，再代入softmax
- 文章说就算有Noisy Top-K Gating还是会有部分专家被过度依赖，因此提出了一个loss，我看loss大致就是把选出来的top k个概率求和，这样梯度下降应该就可以缩小这些概率，达到增强探索的目地，他新增的loss里面还有一些CV之类的不知道是运算符号还是系数，就没有看懂了

## <span id="202502091904"> Scaling Laws </span>
- OpenAI, 2020.1
- Scaling Laws for Neural Language Models
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

## <span id="202503021224"> LoRA </span>
- Microsoft, 2021.6
- LoRA: Low-Rank Adaptation of Large Language Models

<p align = "center">
<img src=/img/lora_training.png width="400" />
</p>

<div align="center">

$$ W = W_0 + \frac{\alpha}{r} B A $$

</div>

- lora就是针对模型的每个权重矩阵学习其两个低秩分解矩阵 r << min(d, k) ，d、k是权重的两个维度，r是B、A的秩，这样需要学习的权重数量就显著下降了，而且也是全参数微调（注意全参微调一般指的就是直接微调所有权重，这里只是说lora也改变了所有参数的权重），lora的灵感来源于[文章1](https://arxiv.org/abs/1804.08838)和[文章2](https://arxiv.org/abs/2012.13255)，大意是说model adaptation learning实际上的参数变化是在一个内在的较低的维度上，所以lora才产生了学习两个低秩分解矩阵的想法
- 上图是在训练期间lora的做法，可以看到和原权重的矩阵乘法是分离的，初始化时对矩阵B使用随机高斯初始化，A直接全部初始化为0，这样在一开始时，ΔW (BA)就是0，alpha人为设置的缩放因子，r是秩
- 在推理时，可以先将lora与权重合并，这样就不需要做两次矩阵乘法了
- GPT说lora通常只微调权重而不微调bias，在原文中对transformer的微调是只微调了attention中的线性变换，没有微调FFN中的
- 在实际使用时（推理）可以再去调alpha，还可以多个lora一起使用（这点很反直觉，应该也没有什么理论依据，明明不是一起训练出来的，却可以一起用），只是多个一起使用需要更仔细地去调每个lora的alpha，平衡每个lora影响的强度
- 后面还诞生了[QLoRA](https://arxiv.org/abs/2305.14314)，就是先对模型做低比特量化（比如4bit, 4-bit NormalFloat, NF4），完了再lora

## <span id="202502191230"> Chain-of-Thought </span>
- Google Research, Brain, 2022.1
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

<p align = "center">
<img src=/img/chain-of-thought_eg.png width="800" />
</p>

- 思维链就是给模型prompt包含一些例子，这些例子首先和问题相关，其次这些例子包含从问题到答案的思考步骤
- 文章说足够大的模型能够在回答问题时复现出思维链思考步骤，从而提高正确率
- 感觉用这个技术的难点显而易见，就是需要人来标注这些精心准备的提示示例，而且还需要提前知晓用户会问什么问题，为每个问题准备好示例

## <span id="202502151904"> Toolformer </span>
- Meta AI Research, 2023.2
- Toolformer: Language Models Can Teach Themselves to Use Tools

<p align = "center">
<img src=/img/toolformer_step.png width="800" />
</p>

- 这篇论文的重点主要是在讲他们是如何构建这个调用API的数据集的，主要包含3个步骤，之后的微调方式和预训练方式无异

<p align = "center">
<img src=/img/toolformer_prompt.png width="400" />
</p>

- 第一步是给模型一个few-shot的prompt，让模型自己标注一下需要使用API的位置，第二步是执行API调用

<p align = "center">
<img src=/img/toolformer_data.png width="400" />
</p>

- 第三步是过滤前两步产生的样本，方法是用weighted cross entropy计算在调用API后的tokens的带权重的对数概率之和（如果不带权重就相当于是后验概率，不理解为什么要加权重，文章也没有说权重怎么设置）取负数L（x1:i−1, e(ci, ri), xi:n，e(ci, ri)是调用API的语句和收到的结果，也就是xi:n这一段tokens），一共要计算三种，第一种是包含完整e(ci, ri)的，第二种是不包含e(ci, ri)的，第三种是API不给回复的，也就是不包含ri的，第二三种取最小值得到L''，令第一种为L'，设置一个阈值tau，只保留L'' - L' > tau的样本，这一通操作也很好理解，简单来说就是只保留调用API对生成后续内容有帮助的样本
- 微调的时候同时需要在上面得到的数据集里加入没有调取API的样本（来自预训练）

## <span id="202502092120"> T5 </span>
- Google, 2019.10
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

<p align = "center">
<img src=/img/T5_task.png width="800" />
</p>

- T5使用传统transformer encoder-decoder的架构，核心思想在于将所有自然语言处理任务都转化为“文本到文本”的格式，能够同时处理机器翻译、文本摘要、问答、情感分析等多种任务
- 模型结构相比原transformer有微小区别，1.移除了原Layer Norm的偏置项，原Layer Norm实现参考torch，2.使用相对位置编码，每个可能的 key–query 位置差都对应一个单一的、可学习的标量，这个标量会被加到注意力计算中，所有层之间，这组相对位置参数是共享的，但在同一层中，不同注意力头各自拥有独立的相对位置偏置，限定了适用的最大相对距离，超出这一范围的所有位置使用相同的 Embedding

<div align="center">

$$ y = \frac{x - \mathcal{E}(x)}{\sqrt{\mathcal{Var}(x) + \epsilon}} \gamma + \beta $$

</div>

- T5使用了teacher forcing和cross-entropy loss，在预训练阶段是像BERT那样mask tokens，但是他是mask完整的一段或多段tokens，而不是像BERT那样非连续的去一个一个地mask token，然后decoder的任务就是输出被mask掉的这些段落，在微调阶段就是标准的像原transformer那样的seq2seq训练
- decoder的生成过程是自回归的，也就是像LLM一样一个token一个token的往外吐，所以才有 'teacher forcing'，所以T5不管是预训练还是finetune都是自回归的，但是BERT那样的encoder模型是一次性预测所有被遮掩的tokens，这值得注意

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

## <span id="202502021741"> InstructGPT </span>
- OpenAI, 2022.3
- Training language models to follow instructions with human feedback

<p align = "center">
<img src=/img/InstructGPT_method.png width="600" />
</p>

- 第二步和第三步实际是迭代进行的，可能会循环多次
- RM (reward model)和PPO policy第一次训练的参数初始化都是来自于SFT (supervised fine-tuning model)，但是PPO value func都是初始化自RM，之和的训练相当于就是持续学习了 (continual learning)，不过只有RM会用到过去收集的数据，PPO不会（这还是遵循on-policy RL的规律）
- 他只用了来自GPT 3的6B模型训练RM，他说175B不稳定
- RM Loss，其中 (k 2) 是组合符号，表示从k个样本中选取两两配对的数量，yw表示相比yl是更符合人类偏好的PPO policy生成的答案，r theta是RM，sigma是逻辑斯谛函数 (logistic function)

<div align="center">

$$ \mathcal{L}(\theta) = -\frac{1}{\binom{k}{2}} \sum_{(x, y_w, y_l) \sim D} \log [\sigma( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) )] $$

</div>

- PPO-ptx Object，其中r_theta是reward model，RL是PPO policy，SFT是supervised fine-tuning policy (model)，beta和gamma是设置的系数，第一项的(x, y)是采样自SFT，第二项的x是采样自预训练数据集，原文表达不严谨，x和y实际都是token序列，并不是一步输出的，是多步输出的序列，所以这里的概率都是条件概率链相乘，如下所示，所以项2相当于是最大化出现这一系列tokens（来自预训练数据）的概率，基本等价于原预训练loss，所以可以认为最大化这个Object，既起到了遵循RM指导的作用（并且不会过于偏离SFT），又起到了遵循预训练数据（分布）的作用

<div align="center">

$$ \mathcal{Object} = E_{(x, y) \sim D_{\text{RL}}}[r_{\theta} - \beta \log (RL(y | x) / SFT(y | x))] + \gamma E_{x \sim {\text{pretrain}}}[\log (RL(x))] $$
$$ RL(x) = \prod_{i=1}^{T} \pi(x_i | x_{<i}) $$
$$ \log(RL(x)) = \sum_{i=1}^{T} \log(\pi(x_i | x_{<i})) $$

</div>

- 这篇文章写的确实相当具有迷惑性，一开始我以为他是把PPO的clip surrogate object改成了以上的object直接优化，但是仔细翻看了多遍，发现这个object真是个十分具有诱导性的说法，他这个object里，第一项作为reward肯定是无异议的，但是根据[原RLHF论文](https://arxiv.org/abs/2009.01325)，他还是个多步强化学习问题，这个reward只在最后一步奖励（但是他的gamma设置等于1），其他步大概率就是0奖励了，所以他才非要有一个value func做信用分配，当然他单步也避免了动作太多，求条件概率链相乘时导致数值溢出，第二项肯定不能作为reward奖励，所以第二项应该是在update参数的时候作为单独的一项与PPO的loss结合（类似于entropy loss），再结合原RLHF论文和[Anthropic RLHF](https://arxiv.org/abs/2204.05862)来看，他第一项中r_theta后面那一项应该是求这两个分布的KL divergence，所以比较合理的做法以Anthropic RLHF的reward作为reward，再把他这个公式中的第二项放在ppo loss里面（当然肯定得取个负号）

## <span id="202502021742"> Claude </span>
- Anthropic, 2022.4
- Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
- 使用了[OOD detection techniques](https://arxiv.org/abs/2106.03004)来让模型回答：“不知道”，这个技术通常被称为ODIN(Out-of-Distribution detector for Neural networks)，它包含两个关键计算1.温度缩放(Temperature Scaling)2.输入扰动(Input Preprocessing)，整个过程简单来说就是让输入softmax之前的logits除以一个温度系数T（通常大于1），缩小概率分布的方差（差距），然后令被选出的类（概率最大的类）的对数概率对于输入x求导，令倒数乘以一个“步长”epsilon（一个较小的正数），然后用于改变x，最后再带入算softmax概率分布（依然要带入T），最大概率值小于阈值，则认为是OOD
- 文章提到他们有部署在线上并且持续循环训练RM和RL policy的步骤，结果显著改善了模型
- 用ELO score训练RM，没看懂这是什么
- reward奖励计算（个人认为原RLHF论文、InstructGPT和这篇论文的这个计算方式应该是一样的，所以InstructGPT的reward计算实现也应该参考这个）

<div align="center">

$$ reward = r_{PM} - \lambda_{KL} D_{KL}(policy || policy_0) $$

</div>

## <span id="202502191442"> DPO </span>
- Stanford University, 2023.5
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model

<p align = "center">
<img src=/img/DPO_RLHFvsDPO.png width="1000" />
</p>

- DPO的loss，其中yw比起yl是更符合偏好的样本，Π theta是要训练的模型，Π ref是不能偏离太远的参考模型（比如SFT），beta是系数用来scale

<div align="center">

$$ \mathcal{L_{DPO}} = -E_{(x, y_w, y_l) \sim D} \big[\log\sigma \big(\beta RATE(y_w) - \beta RATE(y_l) \big) \big] $$
$$ RATE(y) = \log\frac{\pi_{\theta}(y | x)}{\pi_{ref}(y | x)} $$

</div>

- 对DPO的loss求导，可以发现最小化loss时，项2 (Item 2)是在最大化yw的可能性，项3是最小化yl的可能性，项1是隐式的奖励，由KL divergence决定，用来决定最大最小化的强度

<div align="center">

$$ \mathcal{grad} = -\beta E_{(x, y_w, y_l) \sim D} [Item_1 \times (Item_2 + Item_3)] $$
$$ Item_1 = \sigma (\beta RATE(y_l) - \beta RATE(y_w)) $$
$$ Item_2 = \nabla_{\theta} \log \pi (y_w | x) $$
$$ Item_3 = -\nabla_{\theta} \log \pi (y_l | x) $$

</div>

- 文章说实际使用时，你可能会去互联网上下载开源数据，用开源数据的时候，你就没有Π ref了，那你需要initialize Π ref by maximizing likelihood of preferred completions (x, yw)，就是最大化生成符合偏好样本的对数概率之和，我猜他可能是要让你用自己的模型根据开源偏好数据和该优化目标微调一下得到Π ref

## <span id="202502021743"> Llama 1 2 3 </span>
- Meta, 2023.2, 2023.7, 2024.7
- LLaMA: Open and Efficient Foundation Language Models
- Llama 2: Open Foundation and Fine-Tuned Chat Models
- The Llama 3 Herd of Models
- LLaMA和GPT 3一样使用Pre-normalization，也就是normalize transformer block的input而不是output，使用的是[RMSNorm](https://arxiv.org/abs/1910.07467)，首先我们回忆一下为什么要用LayerNorm，[因为神经网络会遇到内部协方差偏移的问题，每层输入的分布会因为前一层网络的参数更新而变](https://arxiv.org/abs/1502.03167)，以下是LayerNorm的计算方式（应该省略了偏置项），a是输入，a bar是输出，i表示向量（张量）中的第i个数值，g是用来重新缩放数值的参数，被初始化为1，LayerNorm被认为有用的主要原因是其缩放不变性和中心化不变性

<div align="center">

$$ \overline{a}_i = \frac{a_i - \mu}{\sigma} g_i $$
$$ \mu = \frac{1}{n} \sum_{i=1}^{n} a_i $$
$$ \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (a_i - \mu)^2} $$

</div>

- RMSNorm认为LayerNorm的中心化不变性没有用，所以他改成了

<div align="center">

$$ \overline{a}_i = \frac{a_i}{RMS} g_i $$
$$ RMS = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2} $$

</div>

- LLaMA使用了[SwiGLU](https://arxiv.org/abs/2002.05202)激活函数替换ReLU，见以下FFN (feed forward network)，Swish 1就是Swish beta，beta等于1，当beta不设置为1时，他是个可学习参数，*代表逐元素乘法（Hadamard乘积），他原文这里用的是外积符号⊗，两个向量的外积结果是矩阵，内积结果是标量，都不合理

<div align="center">

$$ FFN_{ReLU} (x, W_1, W_2) = max(xW_1, 0)W_2 $$
$$ FFN_{SwiGLU} (x, W, V, W_2) = (Swish_1(xW) * xV)W_2 $$
$$ SwiGLU(x, W, V, b, c, \beta) = Swish_{\beta} (xW + b) * (xV + c) $$
$$ Swish_{\beta} (x) = x\sigma(\beta x) $$

</div>

- LLaMA使用了[RoPE (rotary positional embedding)](https://arxiv.org/abs/2104.09864)替换PE (absolute positional embeddings)，RoPE和PE在用法上就有直接的区别，PE是在带入multi-head之前就对输入x编码了，而RoPE是在计算完qkv之后，仅对q和k编码，这也很好理解，因为只有qk才会乘在一起得到权重矩阵，相对位置信息理应只需要保留在这个权重里面，q和k在乘以他们的绝对位置（旋转矩阵）之后，再算注意力权重矩阵时会再相乘，得到的就是相对位置了（公式由GPT生成），代码实现详见[torchtune](https://pytorch.org/torchtune/main/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings)

<div align="center">

$$ (R(\theta_i)Q_i)^\mathrm T (R(\theta_j)K_j) = Q_i^\mathrm T R(\theta_i)^\mathrm T R(\theta_j)K_j $$
$$ R(\theta_i)^\mathrm T R(\theta_j) = R(\theta_j - \theta_i) $$

</div>

- LLaMA使用了[AdamW optimizer](https://arxiv.org/abs/1711.05101)，AdamW和Adam的区别简单来说就是，AdamW通过将权重衰减与梯度更新分离，解决了Adam中权重衰减实现不理想的问题
- LLaMA使用了cosine learning rate schedule，可以用[torchtune的CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)，只是warm-up要自己实现，LLaMA也使用了warm-up，warm-up指的是训练初期使用较小的学习率，并逐步增加到预设的最大学习率，通常是线性或指数增长策略

<div align="center">

$$ lr(t) = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + cos(\frac{\pi t}{T})) $$

</div>

- LLaMA使用了weight decay of 0.1 and gradient clipping of 1.0，weight decay of 0.1就是对模型权重做软约束，在loss中添加模型权重的L2范数正则化项，0.1就是LLaMA使用的这一项前面的系数
- LLaMA借鉴了[causal multi-head attention](https://arxiv.org/abs/2112.05682)和[FlashAttention](https://arxiv.org/abs/2205.14135)的高效实现，实现了xFormers中的[Memory-efficient attention](https://facebookresearch.github.io/xformers/components/ops.html)并用在模型中，通过分块计算和在线累加等方式，优化了内存占用，使模型能够处理更长序列
- LLaMA优化了activations重计算的问题来加速训练，也就是把一些复杂的activations存下来，就不需要在backward的时候再算一遍，就像[这篇文章](https://arxiv.org/abs/2205.05198)中讨论的那样
- LLaMA进行了极少量（和预训练数据相比）的指令微调 (instruction finetuning)来测试模型在MMLU等数据集上的表现，方法参考[这篇论文](https://arxiv.org/abs/2210.11416)，简单来说，像GPT 3中的few-shot learning的例子你如果把它拿来训练模型，这就属于指令微调，InstructGPT中的SFT也属于指令微调

<p align = "center">
<img src=/img/llama_1vs2.png width="750" />
<img src=/img/llama_2step.png width="800" />
</p>

- Llama 2和LLaMA的预训练方式、SFT（指令微调）、模型等基本一样，但是Llama 2使用了更多的数据，提高了context length，使用了grouped-query attention (GQA)，还用了RLHF，不过他和Claude更像，也是去对齐helpfulness and safety
- grouped-query attention (GQA)就是将multi-head多头分成多个组，q是独立的和以前一样，但是同一个组内的kv共享参数，可以节约计算和储存kv cache的开销
- Llama 2发现只需要少量的高质量对话数据用做SFT，就可以让模型学会高质量的对话，比如数万的数量就可以，他们只用了27540个，他们做SFT时，初始学习率降低成0.00002，使用0.1的L2正则，64的batch size和4096的context length，还有2 epochs
- Llama 2训练了2个RM，一个Helpfulness RM，另一个Safety RM，他们请人标注是像Claude那样成对标注的（分成了几个等级，比如好很多，好一点，差不多，差一点），然后训练RM的目标是改进自InstructGPT的目标，y_c是更符合偏好的生成，m(r)是离散的等级差距函数，他们训练RM还结合了开源数据，最后他们以分段函数的形式结合了两个RM，分段的中心思想就是优先关注安全，再是有用

<div align="center">

$$ L = -log(\sigma(r_{\theta}(x, y_c) - r_{\theta}(x, y_r) - m(r))) $$

</div>

- Llama 2使用了两种RL的方法，一种就是PPO，另一种是 Rejection Sampling fine-tuning (RS), 在前几轮里只用RS，后面把PPO用到RS的最优结果上，然后他们最大的模型才用了RS，小的模型都是使用的大模型RS的样本（他们称此为蒸馏 distillation）
- Rejection Sampling fine-tuning (RS) 就是对一个prompt生成多个回答，然后选出RM评分最高的那个，然后用最大化对数概率之和训练（这个是我猜的，因为他毕竟叫fine-tuning，原文又没写）
- Llama 2使用提出了一个叫 Ghost Attention (GAtt)的方法，以用来让模型一直重点关注某些提示，比如扮演成某个名人等，他的做法没有看懂，似乎是不断简洁精炼这些重要的系统提示，然后再与后续的对话拼接在一起？
- 沐神说现在很多llm都是支持的8k上下文，训练的时候上下文是8k，但是部署的时候可以是32k，从实用上，32k的上下文长度对llm就够了，128k就更够了
- 沐神说Llama 3没有给出具体的数据采样方法，就是在什么训练时期，哪些类型的数据（比如数学、code）的采样率是多少，这个数据采样率也十分重要
- Llama 3论文做了比较详细的数据清洗和分类（详见原文），包括各种去重，基于模型的筛选，基于模型的专门从网页提取代码推理数据，基于模型的知识分类，基于scaling laws的最佳知识混合方法，最后得到的比例是50%的通用知识，25%的数学和推理，17%的代码和8%的多语言，以及使用了退火数据
- Llama 3使用了传统transformer (dense transformer)架构，使用了GQA，使用了attention mask去遮罩同一个序列中来自不同文档的tokens，让他们不要互相关注，这个技术在长上下文里至关重要，因为在Llama 3中，对于405B的模型来说，在standard pre-training stage上下文是8k，在continued pre-training stage是128k，使用了RoPE

<p align = "center">
<img src=/img/llama_3size.png width="600" />
</p>

- Llama 3还说他们的budget是3.8*10^25 FLOPs，根据scaling laws和他们的数据量，405B大小是最好的选择，他们认为以前的scaling laws已经不奏效了，原因是以前的测试预算都很小，而且是以预训练数据作为validation loss，他们为此设计了自己的scaling laws，他们首先使用在不同下游任务上的negative log-likelihood作为validation loss，然后做了大量实验得到了以下结果，上左图的不同弧线代表不同固定的总FLOPs (budget)下，过模型的token量（训练token）不同得到的不同结果，上右图得到了最优模型的budget与训练token的关系，他们还预测了（利用之前大量的实验结果和一些旧模型）归一化的negative log-likelihood与预算和正确回答率的关系，下右图的关系是sigmoidal relation

<p align = "center">
<img src=/img/llama_3sl.png width="800" />
<img src=/img/llama_3perfpred.png width="800" />
</p>

- Llama 3用了tensor parallelism (TP), pipeline parallelism (PP), context parallelism (CP), data parallelism (DP)，达到了38-43%的显卡利用率，详见原文
- Llama 3预训练有 (1) initial pre-training, (2) long-context pre-training, and (3) annealing 三个阶段，也用了余弦和warm up，阶段一上下文一开始是4k，一段时间后提升为8k，并且多次调整了数据混合比例，阶段2时他们经过了6次逐步增加上下文长度，最后才到128k，且每次他们都会等模型适应新长度才继续提升，在最后一个阶段，只剩最后少量训练token时，他们分别使用了一些各个领域极高质量的数据，并线性的将学习率降为0，在每个领域分别训练了一个模型，最后再将这些模型权重取平均值，得到最终模型

<p align = "center">
<img src=/img/llama_3posttraining.png width="800" />
</p>

- Llama 3 的 post-training （后训练）阶段有所不同，他们整个后训练也包含多种多样的数据意在提升模型的不同能力，整个后训练会循环6轮，每轮是先训练RM，然后由RM进行RS，然后用RS的最好结果来进行SFT，让你和再进行DPO
- 后训练RM数据收集：他们会先让人类写少量示例，并让上一轮的多个模型（如果是第一轮就是预训练的超参数不同或不同数据混合的多个模型）针对同一个prompt做出多个回答，然后他让标注员选出最差的一个，并标出其他几个相比最差的一个好了多少，同时也允许标注员自己去写一个最好的答案，然后只保留最差的和比最差的好的回答凑成对（“只保留...”这里不确定原文是不是这个意思，有点没看懂，感觉是这样），把这些都拿去训练RM和过去每轮收集的数据都拿去训练RM，RM的loss和Llama 2一样，但是去掉了m(r)，因为他没有分级了
- 后训练SFT：遵循RS，拿上一轮最好的模型生成多个回答，让RM选出最好的一个，然后需要对生成的答案做一些处理（详见原文），再结合一些人类示例数据和为了提升特定能力（比如调用工具）而制作的数据（详见原文，这里包含7种特殊数据，原文很详细），一起进行SFT
- 后训练DPO：用本轮RM处新收集的模型产生的数据来进行DPO，在做DPO时会mask掉特殊头 (various special head)和结束token不让他们参与loss计算，不然模型会胡乱输出结束token，同时在DPO里要加入NLL loss (Negative Log Likelihood)（和InstructGPT一样），最后他们还是像预训练那样去合并多个模型的权重
- Llama 3还尝试了多模态，包括图片和视频输入，以及语音转成文字输入，详见原文

## <span id="202502022356"> Mistral AI Models </span>
- Mistral AI, 2023.10, 2024.1, 2024.10
- 是原LLaMA团队出来创业的成果，[是一系列模型](https://docs.mistral.ai/getting-started/models/models_overview/)，他有的模型没有在arxiv上写文章，只在官方新闻上说了一下，这种模型都没有讲结构，只讲了performance，估计这种和以前的结构一样，只是增大了参数量，换了数据和训练方法，这种以下没有列出来讲
- Mistral 7B
- Mixtral of Experts
- Pixtral 12B
- Mistral 7B结构和LLaMA十分相似，在LLaMA基础上额外使用了GQA、SWA和Pre-fill and Chunking技术
- [sliding window attention (SWA)](https://arxiv.org/abs/2004.05150)的每个token只关注其局部邻域内的一部分token，或是关注固定距离的一部分token，如下所示，在Mistral 7B里是只关注其局部邻域如图中2和下所示（下是Mistral 7B原文示意图），图下3是说经过多层堆叠，注意力影响还是可以蔓延开来

<p align = "center">
<img src=/img/mistral7b_attweight1.png width="800" />
<img src=/img/mistral7b_attweight2.png width="800" />
<img src=/img/mistral7b_attweight3.png width="800" />
</p>

- Pre-fill and Chunking是内存优化技术，Pre-fill比较常见，大部分LLM都会用，就是存kv cache，Chunking感觉应该是配合SWA使用的，因为SWA只需要邻域attention权重，所以可以对过长的prompt分块处理，分块加载入内存（显存）
- 未解决的疑问：[NVIDIA TensorRT-LLM也有分块预填充功能](https://developer.nvidia.com/zh-cn/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/?utm_source=chatgpt.com)，不知道如果没有SWA这个该怎么用？

<p align = "center">
<img src=/img/mixtralofexperts_layer.png width="500" />
<img src=/img/mixtralofexperts_block.png width="700" />
</p>

- Mixtral of Experts和Mistral 7B基本一样，除了FFN都换成了MoE（上图中上面那张），结构比较类似于[Switch Transformers](https://arxiv.org/abs/2101.03961)（上图中下面那张），以下是公式，其中Top2就是Top K选择，SwiGLU_i就是专家，没被选择的那些专家不用参与运算，注意Mixtral of Experts就像Switch Transformers一样，针对每个token都有选择不同专家的过程

<div align="center">

$$ y = \sum_{i=0}^{n-1} Softmax[Top2(x \times W_g)] \times SwiGLU_i(x) $$

</div>

- 因为只选择Top K个专家，没被选择专家不运算，所以会中断从没选到的专家那里传来的梯度，之前有个误区就是以为这里会中断梯度回传到gate网络，实际不会，只让参与运算的专家回传即可，但是正是因为这个问题会导致MoE的训练不稳定（比如长期某个专家未被选到，突然被选到了）
- Mixtral of Experts有一个疑点就是在训练阶段一般MoE都会去平衡对专家的选择，通常是添加Auxiliary Loss，Mixtral of Experts虽然没提，但是大概率是要添加的，添加该loss会产生新问题，比如在Switch Transformers中，loss如下，其中f表示每个专家实际被选中的token比例（离散计数不可微）P表示每个专家softmax后的概率之和（可微），这里在backward时就需要使用直通估计器 (Straight‑Through Estimator, STE)，做法就是将上一个倒数直接作为不能求导的环节的倒数，比如对于 ∂L/∂x=∂L/∂y⋅∂y/∂x，因为∂y/∂x不能求，所以直接将 ∂L/∂y 作为 ∂L/∂x 的估计，相当于是将f直接视为常数（例外GPT还提到了一种解决类似问题的方法叫Gumbel-Softmax，似乎在离散DDPG处看到过，之后看一看）

<div align="center">

$$ L_{load} = \alpha N \sum_{i=1}^{N} f_i P_i $$

</div>

- 听说Mistral Large 2比Llama3.1擅长代码和数学，大小是123B
- Pixtral 12B是一个多模态模型，可以接收多轮对话的文字和图片，可以处理128k上下文，是由预训练好的transformer encoder + transformer decoder微调而来，它由一个专门处理图像的encoder叫作Pixtral-ViT (400M)和一个表现最好的Mistral Nemo 12B （这个模型应该结构和Mistral 7B一样，但是训练的可以处理128k上下文）作为decoder组成

<p align = "center">
<img src=/img/pixtral12b_encoder.png width="800" />
</p>

- encoder需要被训练的可以适应不同的分辨率和长宽比，其结构源于ViT，只是做了4个改动，1.Break tokens，在图像换行处加入[IMAGE BREAK]，在图像结尾加入[IMAGE END]，2.FFN中加入GLUE家族激活函数，3.加入block-diagonal mask让不是一张图片的之间没有注意力权重，4.用了RoPE-2D，这是他们改进自RoPE的算法，旋转矩阵没有学好，详见原文，文章没有具体提是怎么预训练encoder的，但是引用了CLIP

<p align = "center">
<img src=/img/pixtral12b_decoder.png width="800" />
</p>

- 图中那个Vision-Language Projector是一个用了GeLU激活的两层fcn，用来统一dimension，在decoder中图像token也会被视为文字token，比如同样要使用1D的RoPE处理，文章也没有说encoder和decoder合并后，要怎么一起微调

## <span id="202503302359"> DeepSeek Models </span>
- DeepSeek-AI, 2024.1, 2024.1, 2024.2, 2024.3, 2024.5, 2024.12, 2025.1
- [是一系列模型](https://www.deepseek.com/)
- DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
- DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- DeepSeek-VL: Towards Real-World Vision-Language Understanding
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- DeepSeek-V3 Technical Report
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- DeepSeek LLM 的结构就是LLaMA的结构，但是较大的模型67B使用了GQA，一样使用了warm up的余弦调度，其他超参数详见原文，服务器架构使用了幻方量化自研的[HAI-LLM](https://www.high-flyer.cn/en/blog/hai-llm/), 就像[Megatron](https://github.com/NVIDIA/Megatron-LM)那样融合了data parallelism, tensor parallelism, sequence parallelism, and 1F1B pipeline parallelism, 使用[flash attention](https://github.com/Dao-AILab/flash-attention)提高硬件利用率, [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)的ZeRO-1进行优化器计算优化，DeepSeek LLM还研究了scaling laws（详见原文），并使用了SFT和DPO

<p align = "center">
<img src=/img/DeepSeekMoE_moe.png width="800" />
</p>

<div align="center">

$$ h_t^l = \sum_{i=1}^{K_s} FFN_i(u_t^l) + \sum_{i=K_s+1}^{mN} \big( g_{i, t} FFN_i(u_t^l) \big) + u_t^l $$
$$ 
    g_{i, t} =
    \begin{cases}
        s_{i, t}, s_{i, t} \in Topk( \{ s_{j, t} | K_s + 1 <= j <= mN \}, mK - K_s), \\
        0, otherwise, 
    \end{cases}
$$
$$ s_{i, t} = Softmax({u_t^l}^T \times e_i^l) $$
$$ L_{ExpBal} = \alpha_1 \sum_{i=1}^{N^{\prime}} f_i P_i, N^{\prime} = mN - K_s $$
$$ f_i = \frac{N^{\prime}}{K^{\prime} T} \sum_{t=1}^{T} \pmb{1}(t), K^{\prime} = mK - K_s $$
$$ P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i, t} $$
$$ L_{DevBal} = \alpha_2 \sum_{i=1}^{D} f_i^{\prime} P_i^{\prime} $$
$$ f_i^{\prime} = \frac{1}{|\epsilon_i|} \sum_{j \in \epsilon_i} f_j, \epsilon_i \in \{ \epsilon_1, \epsilon_2, ..., \epsilon_D \} $$
$$ P_i^{\prime} = \sum_{j \in \epsilon_i} P_j $$

</div>

- DeepSeekMoE 和通常的MoE架构区别不大，简单来说就是增加了多个共享专家是每个token都要路由的，且把专家数量翻了m倍（比如N个变成mN个，原本每次选top K个也变成选top mK个），且每个专家参数量也缩小，所以总参数量还是不变（上图a是通常，bc是DeepSeekMoE），因为作者认为这样有利于更多的可能专家组合，且每个专家变小了，知识特化会做的更好，在load balance上使用了两个loss，一个是 Expert-Level Balance Loss（公式456），一个是 Device-Level Balance Loss（公式56789），第一个loss就是专家负载均衡，第二个是设备负载均衡，作者给第一个赋的权重alpha 1较小，第二个alpha 2较大，1(t)是指示函数，当token t选择专家i时等于1否则为0，有D个计算集群，每个集群里面有|epsilon_i|个设备
- DeepSeekMath 微调自[DeepSeek-Coder](https://arxiv.org/abs/2401.14196) 7B，超过一半数据来自 Common Crawl (CC) ，使用部分[OpenWebMath](https://arxiv.org/abs/2310.06786)作为正例部分CC作为负例，训练了一个[fastText model](https://arxiv.org/abs/1612.03651)，使用该模型从CC中提取 mathematical web pages ，还对提取的数据质量进行了验证（详见原文），SFT数据是经过领域和难度划分的，problems are paired with solutions 通过 chain-of-thought (CoT), [program-of-thought (PoT)](https://arxiv.org/abs/2211.12588), [tool-integrated reasoning format](https://arxiv.org/abs/2309.17452)，获得了776K的 training examples

<p align = "center">
<img src=/img/deepseek_mathPOT_1.png width="400" />
<img src=/img/deepseek_mathPOT_2.png width="400" />
</p>

- PoT就是将思维链过程写成一步一步的python代码，然后用python解释器求解，因为在代码中间变量也可以用符号表示，不需要立刻求解，这样子就避免了CoT中自然语言求解数值问题时易发生的错误

<p align = "center">
<img src=/img/deepseek_mathtirf_1.png width="600" />
<img src=/img/deepseek_mathtirf_2.png width="600" />
</p>

- tool-integrated reasoning format认为单单是CoT的文字思维链和单单是PoT的写代码都是行不通的，应该将这个过程组合起来，因此它的推理过程是 qpraoraorao...rao..., q就是提出的问题，p是给的示例提示，如果是zero-shot，p就没有，r是文字思维链，a是写的要给解释器的代码，o是解释器返回的结果，然后直到达到最大rao循环数量或模型输出终止符，即停止，论文是用GPT 4生成的这个过程数据，并用模仿学习 (minimizing negative log-likelihood loss) 去训练他们的模型，还有一个Output Space Shaping步骤来对模仿学习后的模型再训练纠错（详见原文）

<p align = "center">
<img src=/img/deepseek_mathgrpo_diagram.png width="800" />
</p>

<div align="center">

$$ \mathcal{Object}_{GRPO}(\theta) = E_{q \sim P(Q), \{o_i\}_{i = 1}^G \sim \pi_{\theta_{old}}(O | q)} \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \big\{ min \big[ R \hat{A}_{i, t}, clip \big( R, 1 - \epsilon, 1 + \epsilon \big) \hat{A}_{i, t} \big] - \beta D_{KL}[\pi_{\theta} || \pi_{ref}] \big\} $$
$$ R = \frac{\pi_{\theta}(o_{i, t} | q, o_{i, < t})}{\pi_{\theta_{old}}(o_{i, t} | q, o_{i, < t})} $$
$$ D_{KL}[\pi_{\theta} || \pi_{ref}] = \frac{\pi_{ref}(o_{i, t} | q, o_{i, < t})}{\pi_{\theta}(o_{i, t} | q, o_{i, < t})} - log \frac{\pi_{ref}(o_{i, t} | q, o_{i, < t})}{\pi_{\theta}(o_{i, t} | q, o_{i, < t})} - 1 $$
$$ \hat{A}_{i, t} = \widetilde{r}_i = \frac{r_i - mean(\pmb{r})}{std(\pmb{r})}, \pmb{r} = \{ r_1, r_2, ..., r_G \} $$
$$ \hat{A}_{i, t} = \sum_{index(j) >= t} \widetilde{r}_i^{index(j)} $$
$$ \widetilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - mean(\pmb{r})}{std(\pmb{r})}, \pmb{r} = \big\{ \{ r_1^{index(1)}, ..., r_1^{index(K_1)} \}, ..., \{ r_G^{index(1)}, ..., r_G^{index(K_G)} \} \big\} $$

</div>

- 注意，因为 DeepSeekMath 已经经过CoT、PoT等数据类型的SFT了，现在模型的输出形式已经变化了，会输出类似 [STEP] 这样的特殊类型token作为分隔符，将整个回答分成 [STEP]...[STEP] 一段一段的推理过程，每一段被视为一个推理步 (reasoning step)，当然每一个时间步还是一个token
- DeepSeekMath 最后使用了 Group Relative Policy Optimization (GRPO) 进行训练，GRPO需要最大化以上目标，GRPO和PPO的区别如下，GRPO没有critic，因此模型大小相当于小了一半，且RLHF的奖励信号是稀疏的，因为只有最后一个推理步的最后一个token才给奖励，其他时间步奖励都是0，作者说这样子增加了训练critic的难度，所以作者认为没有critic训练变简单了占用资源变小了，所以GRPO就需要解决预测状态价值baseline的问题，解决方法就是 "Group" ，只要我每次不是只采样一个 trajectory ，而是采样 G 个，那我取“平均值”不就相当于有baseline了撒，所以在 Outcome Supervision RL with GRPO（稀疏奖励信号，仅最后一个推理步的最后一个token有奖励）的设置下，估算 advantage value (adv) 的方法就是normalization G个reward，然后直接认为整条trajectory的每一个时间步的adv都是刚刚算出的这个normalization reward（公式4），另外作者还讨论了一种 Process Supervision RL with GRPO 的设置，是他们的另一篇论文 [MATH-SHEPHERD](https://arxiv.org/abs/2312.08935) 的成果，能获得每一个推理步的最后一个token的奖励，虽然也是稀疏的，但是可以让模型更易区分每个推理步的对错，因此adv的计算方式被改为了公式5、6，注意index(j)是第j个推理步的最后一个时间步 (token) 的index，K自然就是最后一个推理步，求解该adv简单来说，就是认为每一个推理步内的时间步的adv是一样的，adv(t)等于时间步t所在的推理步及其之后的推理步的index(j)（推理步内的最后一个时间步）的normalization rewards之和
- 还可以看到 GRPO 有一个KL divergence的项类似于RLHF中奖励数值里要加入的KL项让policy不要被训练得太偏离参考模型（通常是SFT后的模型），但是他们改进了该项，换成了Schulman（提出PPO的那位大牛）推导出的对KL divergence的无偏估计，详细可参考[blog](http://joschu.net/blog/kl-approx.html)

<p align = "center">
<img src=/img/deepseek_mathgrpo_pseudocode.png width="800" />
</p>

- 看伪代码可以看到每次更新reward model (RM)时参考模型也会更新成policy model (PM)，更新RM是继续训练旧的RM，数据集是 based on the sampling results from the policy model and incorporates 10% of historical data，合成训练RM的数据集方法也是来自 MATH-SHEPHERD ，另外PM是初始化自 DeepSeekMath-Instruct 7B ，RM是训练自DeepSeekMath-Base 7B with a learning rate (lr) of 2e-5，训练 PM 超参数 lr 1e-6, beta (KL coefficient) 0.04, G 64, batch size 1024, and only has a single update following each exploration stage，最后发现只用CoT数据训练RL都能有较大的提升，且代码数据有利于数学推理，arxiv数据没什么用，最后作者将SFT, Rejection Sampling Fine-tuning (RS), DPO, Online RS, PPO, GRPO都统一成了一种泛RL范式（详见原文），作者通过实验结论认为RL相比SFT是提升了模型的最优表现，而SFT是提升基础能力

<p align = "center">
<img src=/img/deepseek_math_math-shepherd.png width="800" />
</p>

- MATH-SHEPHERD就是像蒙托卡罗树搜索那样，从每一个推理步开始或许会再往后采样若干个trajectory，然后分为两种估计模型，硬估计和软估计，硬估计是往后的trajectory只要有一条能得到正确答案，则该推理步打分1，软则是正确数量占比作分数

<p align = "center">
<img src=/img/DeepSeek-VL_task.png width="1000" />
<img src=/img/DeepSeek-VL_training.png width="1000" />
</p>

- DeepSeek-VL 是个vision language model，他处理的任务如上图1，是一张图片加一句提示词来给回答，不是视觉语言多轮对话任务，比较简单，数据混合了视觉语言数据和DeepSeek LLM用的纯语言数据（详见原文），模型由一个混合视觉编码器 (hybrid vision encoder)，一个视觉适配器 (vision adaptor) 和 语言模型 (language model) 组成，其中混合视觉编码器结构没讲清楚，只表明了使用两个预训练模型，一个 [SAM-B](https://arxiv.org/abs/2304.02643) 一个 SigLIP-L 没有给出引用，猜想是[该模型](https://arxiv.org/abs/2303.15343)，经过一些处理后，两个模型的输出会被连接在一起，视觉适配器就是MLP，语言模型就是DeepSeek LLM，训练包含三个阶段，如上图2，冰花表示冻结参数，火焰表示打开训练
- DeepSeek-V2 是 MoE 模型，总大小236B，每个token激活21B，DeepSeek-V2 使用 Multi-head Latent Attention (MLA) 替换 GQA，用 DeepSeekMoE 的 MoE （替换 FFN ），用了[YaRN](https://arxiv.org/abs/2309.00071)在初始预训练之后增长上下文窗口从4k到128k（详见YaRN论文）

<p align = "center">
<img src=/img/deepseek_v2_mla.png width="800" />
</p>

- MLA 是为了解决 kv cache 占用太多储存空间的问题，MLA用了和lora一样的低秩矩阵分解的思想，先用一个下投影矩阵将 hidden states（就是MHA里需要乘以3个投影矩阵得到qkv的那个输入张量） 压缩成 low-rank key-value joint compression，就不需要储存kv cache了，存该压缩张量就行了，之后再有两个上投影矩阵将其恢复成kv就行了，只是需要注意的是这令 RoPE 变得不兼容，一般 k cache 是储存的经过RoPE后的k，然而在MLA中这显然不行（上图画的不准确，实际对q他们也用了低秩矩阵分解），为了防止每次恢复出k之后再带入算之前所有token算RoPE，他们提出了 Decoupled Rotary Position Embedding（对q也一样操作），简单来说就是准备了两份q和k，对于q来说，先压缩得到压缩张量（hidden states乘以q的下投影矩阵），对该张量分别乘以两个矩阵，一个得到不带位置信息的q，另一个再代入RoPE得到带位置信息的q，再concatenate，对于k来说，由hidden states乘以两个k的投影矩阵得到两个k，一个是下投影矩阵，得到的结果乘以2个上投影矩阵得到kv，另一个是一个独立的投影矩阵的，该结果再代入RoPE得到带位置信息的k（这一步和普通的RoPE对k的操作是一样的），然后concatenate，因此除了需要缓存kv压缩张量，还需要缓存带位置信息的k，注意为了使运算更快，那两个参与得到带位置信息的qk的投影矩阵投影的维度和下投影矩阵一样都很小

<div align="center">

$$ L_{CommBal} = \alpha_3 \sum_{i=1}^{D} f_i^{\prime\prime} P_i^{\prime\prime} $$
$$ f_i^{\prime\prime} = \frac{D}{MT} \sum_{t=1}^{T} \pmb{1}(t) $$
$$ P_i^{\prime\prime} = \sum_{j \in \epsilon_i} P_j $$

</div>

- DeepSeek-V2 在 DeepSeekMoE 的 MoE 基础上还增加了 Communication Balance Loss，M是设备数量，D是集群数量，T是token数量，1(t)是指示函数，当token t选择设备i时等于1否则为0，为了负载均衡，还使用了 Token-Dropping Strategy，这个方法是说设备在达到容量时会舍弃亲和力低的token，实在是无法理解token为什么能被舍弃
- DeepSeek-V2 的RL阶段和DeepSeekMath一样，但是是Outcome Supervision RL with GRPO的设置，有两个阶段，第一个是 reasoning alignment stage，第二个是 human preference alignment stage，第一阶段只有一个奖励模型，收集训练奖励模型的数据code来自编译器，数学来自 ground-truth label，第二阶段有三个奖励模型helpful、safety和rule，他们会被人为赋予的权重并求和作为奖励，训练数据是人为精心收集，训练奖励模型不是Point-wise Loss就是Pair-wise Loss，point就是直接赋予数值然后算mse，pair就是InstructGPT的RM loss

<div align="center">

$$ h_t^{\prime} = u_t + \sum_{i=1}^{N_s} FFN_i^{(s)}(u_t) + \sum_{i=1}^{N_r} g_{i, t} FFN_i^{(r)}(u_t) $$
$$ g_{i, t} = \frac{g_{i, t}^{\prime}}{\sum_{j=1}^{N_r} g_{j, t}^{\prime}} $$
$$ 
    g_{i, t}^{\prime} =
    \begin{cases}
        s_{i, t}, s_{i, t} + b_i \in Topk( \{ s_{j, t} + b_j | 1 <= j <= N_r \}, K_r), \\
        0, otherwise, 
    \end{cases}
$$
$$ s_{i, t} = Sigmoid({u_t}^T \times e_i) $$

</div>

- DeepSeek-V3 和 DeepSeek-V2 有一些区别，在load balance上额外使用了 [auxiliary-loss-free load balancing strategy](https://arxiv.org/abs/2408.15664)，改了MoE的专家亲和得分 (affinity scores)（他将g称为亲和得分）计算方式，主要是softmax改成sigmoid并进行了归一化，并在专家选择时加入bias（上式的b_i，这就是auxiliary-loss-free load balancing strategy技术），注意这个是不影响求和g的，只在选top k时使用，还有一个超参数 bias update speed gamma，在每一步结束时，overloaded的expert的bias会减去gamma，underloaded的会加，比起V2的很多辅助损失 (auxiliary loss)，V3仅有一个辅助损失 Complementary Sequence-Wise Auxiliary Loss，如下式，1(t)是指示函数，当token t选择专家i时等于1否则为0

<div align="center">

$$ L_{Bal} = \alpha \sum_{i=1}^{N_r} f_i P_i $$
$$ f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \pmb{1}(t) $$
$$ P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i, t}^{\prime} $$
$$ s_{i, t}^{\prime} = \frac{s_{i, t}}{\sum_{j=1}^{N_r} s_{j, t}} $$

</div>

- DeepSeek-V3 使用了 Node-Limited Routing 技术，在 MoE 模型中，每个 token 会被路由到多个专家（expert），而这些专家往往分布在不同的计算节点上，如果一个 token 被路由到过多的节点，会导致大量的跨节点数据交换，严重拖慢训练速度，Node-Limited Routing 确保每个 token 只会被发送到有限数量的节点（例如，最多 4 个节点），具体来说，系统会根据分布在各节点上的专家的亲和分数（affinity scores），为每个 token 计算出每个节点的总亲和分数，并只选择总分最高的几个节点进行路由，这样可以大幅减少跨节点通信，从而实现计算与通信的重叠，极大提高了训练效率（GPT说的），这个技术V2也用了，只是V2是限制设备，所以在V2里叫 Device-Limited Routing（在V2里漏写了），***只是这里就产生疑问了，所以是先限制了节点（设备）的选择，再选top k专家吗？*** V3没有使用 Token-Dropping Strategy

<p align = "center">
<img src=/img/deepseek_v3_multitokenprediction.png width="800" />
</p>

- DeepSeek-V3 使用了 Multi-Token Prediction 技术，该技术源于[论文](https://arxiv.org/abs/2404.19737)，原 Multi-Token Prediction 论文是4 tokens target，DeepSeek-V3没说，但看上图应该也是4个，图画的很清晰，最终loss就是图中 L_MTP_1, L_MTP_2, ... 等MTP loss取平均值再乘一个lambda权重，再与L_main求和，L_main就是普通的next token prediction loss，L_MTP_k就是负对数概率取平均

<p align = "center">
<img src=/img/deepseek_v3_mixedprecisionframework.png width="800" />
</p>

- DeepSeek-V3 的计算设施和并行策略详见原文3.1，3.2章节，DeepSeek-V3 还是用了带有 FP8 的 Mixed Precision Framework，上图是 Linear operator 的计算流程，总的来说 most compute-density operations are conducted in FP8, while a few key operations are strategically maintained in their original data formats 以及一些其他的提升精度的方法，和一套针对高性价比ai设施的建议方案（详见原文章节3后续）
- DeepSeek-V3 的预训练数据处理包括处理冗余，检查完整性以得到高质量数据，对10%的数据使用 Prefix-Suffix-Middle (PSM) framework，以及使用特殊token，SFT 写的感觉不清楚，SFT 和 RL 阶段似乎是混合进行的，Reasoning Data 的few-shot部分是 DeepSeek-R1 生成的（但是DeepSeek-R1却比V3晚发布...），为了平衡R1的高正确率和回答的简洁性和规范化，他们训练了专家模型去生成数据，包含原始回答和R1回答，专家模型给V3生成数据，让V3学会两种回答风格，RL包含两种奖励模型，rule-base和model-base，rule就是数学问题和代码问题，数学可以看其最终答案是否正确，代码可以用编译器验证，model就是其他问题，V3和V2一样，没有过程奖励，只有最终奖励

<p align = "center">
<img src=/img/deepseek_r1_template.png width="800" />
</p>

- DeepSeek-R1 都是从 DeepSeek-V3-Base 训练得到的，训练 DeepSeek-R1-Zero 没有使用SFT（冷启动）数据，只是zero有 poor readability, and language mixing 等问题，所以他们又推出了加上SFT并经过两轮训练的 DeepSeek-R1，训练zero连 neural reward model 都没有用，就只用了像V3一样的rule-base奖励（数学和代码），以及格式奖励，就是让模型用[think]...[/think]特殊token将思考过程包起来[answer]...[/answer]包回答，然后奖励也不是过程奖励，是只奖励最终答案，然后模型都是按以上模板提示的，随着训练进行，推理时长会增加很多（几百到12000），训练R1是为了解决两个疑问：1.通过将少量高质量数据纳入冷启动，可以进一步提高推理性能或加速收敛吗，2.我们如何训练一个用户友好的模型，该模型不仅产生清晰且连贯的思维链，而且还表现出强大的通用能力
- DeepSeek-R1 是两阶段的 SFT + RL 训练，首先收集了一些模型生成人类修改的冷启动CoT数据，这些数据必须具备可读性并符合人类先验模式（不理解什么意思），然后利用zero的训练方式训练RL并额外提供 language consistency reward，第二阶段收集了书写，角色扮演和其他通用任务中的数据和其他非推理数据（有些问题不需要思考就能回答，比如和你打招呼等），也用了 Pair-wise Loss 训练的 neural reward model 和 rejection sampling，也针对 helpfulness and harmlessness 进行了训练

## <span id="202502151055"> Flamingo </span>
- DeepMind, 2022.4
- Flamingo: a Visual Language Model for Few-Shot Learning

## <span id="202502021744"> Whisper </span>
- OpenAI, 2022.12
- Robust Speech Recognition via Large-Scale Weak Supervision

## <span id="202502030008"> Noise2Music </span>
- Google Research, 2023.2
- Noise2Music: Text-conditioned Music Generation with Diffusion Models
- [paper](https://arxiv.org/abs/2302.03917)，[blog](https://google-research.github.io/noise2music/)

## <span id="202502021746"> AlphaFold 1 2 3 </span>
- DeepMind

## <span id="202502021747"> ViLT </span>
- Korea NAVER AI, 2021.2
- ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision

## <span id="202502021748"> ALBEF </span>
- Salesforce Research, 2021.7
- Align before Fuse: Vision and Language Representation Learning with Momentum Distillation

## <span id="202502021749"> VLMo </span>
- Microsoft, 2021.11
- VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts

## <span id="202502021750"> BLIP </span>
- Salesforce Research, 2022.1
- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

## <span id="202502021751"> CoCa </span>
- Google Research, 2022.5
- CoCa: Contrastive Captioners are Image-Text Foundation Models

## <span id="202502021752"> BEiT-3 </span>
- Microsoft Corporation, 2022.8
- Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks

## <span id="202502021745"> DALL-E 1 2 3 </span>
- OpenAI, 2021.2, 2022.4, 2023.8
- Zero-Shot Text-to-Image Generation
- Hierarchical Text-Conditional Image Generation with CLIP Latents
- [Improving Image Generation with Better Captions](https://cdn.openai.com/papers/dall-e-3.pdf)

## <span id="202503051357"> U-Net </span>
- University of Freiburg, Germany, 2015.5
- U-Net: Convolutional Networks for Biomedical Image Segmentation

<p align = "center">
<img src=/img/unet_backborn.png width="800" />
<img src=/img/unet_overlaptile.png width="800" />
</p>

- 上图都画得很清楚，模型有编码器(Contracting Path)和解码器(Expansive Path)组成，最后输出层是(1, 1)卷积，全模型没使用linear transformation，编码器的每个块由2个(3, 3)的卷积（无padding和stride）和一个(2, 2)的max pooling组成，解码器的每个块由1个(2, 2)的反卷积（上采样）和2个(3, 3)的卷积（无padding和stride）组成，另外因为是用来做医学图像分割，所以最后输出的通道是2（2分类任务）
- 解码器中每个块上采样完之后，会与编码器中对应的块的卷积结果（先裁剪4边到适合解码器的尺寸(crop)）concatenate到一起，这样做论文说是为了加强高分辨率细节
- 看上图1可以看出最后输出比输入的长宽小，所以论文利用overlap tile就是把分割图像先分割成重叠的若干份，最后再拼接起来，如果是边界位置就用镜像扩展，这样保证了与label的对应关系（因为label与输入尺寸一样）
- 训练的时候加了个weight map（详见原文），用来提到细胞间分界线这些像素在loss里面的权重，用来让模型更关注细胞边界

## <span id="202503131226"> VQ-VAE </span>
- DeepMind, 2017.11
- Neural Discrete Representation Learning
- 先回忆一下VAE，VAE (Variational Autoencoder) 是一种生成模型，目标是学习数据的潜在分布，从而生成与训练数据相似的新样本，与传统自编码器不同，VAE 不仅学习如何重构输入数据，还通过概率建模得到数据生成的隐变量分布，整个模型由编码器和解码器组成，编码器(Encoder)将输入数据映射到隐变量空间，并输出隐变量的分布参数（通常假设为高斯分布），即均值 μ 和对数方差 log(σ²)（σ是标准差），解码器(Decoder)从隐变量中采样，再将样本映射回数据空间，重构出与原始数据尽量相似的结果
- VAE假设数据x是由隐变量z生成的，即存在生成过程p(x∣z)以及先验分布p(z)（通常假设为标准正态分布），由于直接求解后验分布p(z∣x)通常是不可行的，VAE 用一个近似分布q(z∣x)（由编码器输出）来替代，我们可以用下界ELBO (Evidence lower bound)作为优化目标（式1），需要最大化ELBO，因该形式不易最大化，替换为以下等价形式（式2），其中第一项为重构项，第二项为KL 散度，保证编码器输出的隐变量分布尽量接近先验分布p(z)，由于直接从q(z∣x)中采样会阻断梯度的传递，因此使用重参数化技巧模拟采样：编码器输出均值μ(x)和对数方差log(σ²(x))，标准差σ(x) = exp(log(σ²(x)) / 2)，生成一个标准正态分布噪声 𝜖 ∼ N(0,I)，最终z = μ(x) + σ(x) * 𝜖 ，实际损失函数：实际中将 -ELBO 作为损失函数，因此需要最小化重构损失，最大化KL，重构损失常用MSE（过去的二值图像用BCE (binary cross entropy)），对于高斯分布 q(z∣x) = N(μ, σ²) 和先验分布 p(z) = N(0, I)，KL 散度有解析解（式3），其中d是z的维度，下面是GPT实现的VAE的实现（用的BCE）

<div align="center">

$$ ELBO = ln[p(x)] - D_{KL}(q(z|x) || p(z|x)) $$
$$ ELBO = E_{z \sim q(z|x)}\big[ln[p(x|z)]\big] - D_{KL}(q(z|x) || p(z)) $$
$$ KL = - \frac{1}{2} \sum_{i=1}^{d} [1 + ln(\sigma_i^2) - \mu_i^2 - \sigma_i^2] $$

</div>

```python
import torch
import torch.nn as nn

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # 编码器：输入 -> 隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 分支输出潜在变量的均值和对数方差
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值 μ
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 对数方差 log(σ^2)

        # 解码器：潜在变量 -> 隐藏层 -> 重构图像
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        """
        编码过程：将输入数据映射到潜在空间，输出均值和对数方差
        """
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：通过均值和对数方差采样潜在向量
        """
        std = torch.exp(0.5 * logvar)   # 计算标准差
        eps = torch.randn_like(std)     # 生成与 std 同形状的正态分布噪声
        return mu + eps * std           # 返回采样结果

    def decode(self, z):
        """
        解码过程：将潜在向量映射回原始数据空间
        """
        h3 = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 输出采用 Sigmoid 激活，保证值在 0~1 范围内

    def forward(self, x):
        """
        前向传播过程：包含编码、重参数化和解码三个步骤
        """
        mu, logvar = self.encode(x.view(-1, 784))  # 将图片展平为一维向量
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    定义 VAE 的损失函数：
    1. 重构损失：使用二值交叉熵衡量重构图像和原图像的差异
    2. KL 散度：衡量编码器输出的潜在分布与标准正态分布之间的差异
    """
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

<p align = "center">
<img src=/img/vqvae_entirety.png width="800" />
</p>

- VAE存在后验坍塌 (posterior collapse) 的问题，也就是过于强大编码器会忽略先验信息 (z) ，为了解决这个问题，VQ‑VAE (Vector Quantized Variational AutoEncoder) 将VAE和 VQ (Vector Quantization)相结合，因此 VQ‑VAE 包含编码器、离散隐变量表 (Codebook) 和解码器三个主要部分，在VAE中先验分布和后验分布都被假设为高斯分布，在VQ‑VAE都是离散分布且是被学习出来的
- Codebook有随机初始化的k个向量，编码器的输出通过算欧式距离，取最近的离散向量替换作为解码器的输入，注意是每一个图像的patch对应一个离散向量，因为这里没有梯度回传，所以使用STE，loss由重建损失，Codebook对齐损失和 commitment loss三项组成，其中beta是设置的系数，sg表示梯度停止，z_q(x)就是被选出的e（离散变量），|| ||_2表示L2范数，第一项即重建损失（同VAE），第二项是因为STE不会让梯度流向e，所以加入使得e能够学习并与z尽量靠近，第三项是因为当e没有z学习速度快时会导致其无限增长，所以限制z与e数值接近

<div align="center">

$$ L = log \big[ p[x|z_q(x)] \big] + || sg[z_e(x)] - e ||_2 + \beta || z_e(x) - sg(e) ||_2 $$

</div>

## <span id="202503071100"> VQ-GAN </span>
- CompVis, 2020.12
- Taming Transformers for High-Resolution Image Synthesis

<p align = "center">
<img src=/img/vqgan_entirety.png width="800" />
</p>

- 用VQ-GAN来代指这篇论文肯定是不对的，不过VQ-GAN是这篇论文提出的最重要方法，另外一部分是transformer
- VQ-GAN原文写的有点晦涩难懂，不过他基本上改进自VQ-VAE，首先将原本的encoder、codebook、decoder作为了generator，里有一个patch-based discriminator，在loss上首先将VQ-VAE的重建loss改成了感知损失（perceptual loss，以下用L_rec指代），VQ-VAE loss的其他部分保持不变（以下L_VQ代指这个VQ-VAE loss改），感知损失就是找一个训练好的预训练模型，通常用VGG，然后把你生成的x和原x带入该模型，将该模型的所有中间层输出拿出来，算他们的L2距离，然后加权求和（权重自拟），另有一个判别器loss就是常见的2分类loss，以L_GAN指代，以下是生成器的loss，那个grad G_L指的是选中的生成器的层的对应损失的梯度，通常选输出层，delta是个极小值

<div align="center">

$$ L = L_{VQ} + \lambda L_{GAN} $$
$$ \lambda = \frac{\nabla_{G_L} (L_{rec})}{\nabla_{G_L} (L_{GAN}) + \delta} $$

</div>

- transformer就是在encoder和codebook编码完后，像GPT一样用transformer decoder only的形式进行next token的预测学习，在学习时可以学习带条件 (conditional) 和不带的情况 (unconditional) 

## <span id="202503051346"> Stable Diffusion 1, SDXL, SD 3 </span>
- CompVis, Runway ML, Stability AI, 2021.12, 2023.7, 2024.3
- High-Resolution Image Synthesis with Latent Diffusion Models
- SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
- SD 1和VQ-GAN + transformer的结构可以说差不多，可以说就是对VQ-GAN做了小修改，并且把transformer的predict next token换成了U-Net diffusion
- SD 1对VQ-GAN的小修改就是增强其avoid arbitrarily scaled latent spaces的能力（VQ-VAE loss第三项就是在做这件事），有两种方法解决这个，只选其中一种，1.增加VAE loss中的KL divergence项（使用极小的权重，原文是10^(−6)），2.增大codebook的dimension
- SD 1的Latent Diffusion Models就是[DDPM (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239)，2020年发表的DDPM被认为是带火扩散模型的关键，DDPM的建模过程有两个阶段，一个是正向扩散，另一个是逆向生成，两个阶段都是马尔可夫链
- DDPM正向扩散从原始图像x0开始，每一步都向图像中添加少量高斯噪声，得到x1, ...xT，当T足够大时xT将接近于标准正态分布，即纯噪声，式1是单步过程，式2是从x0直接采样，注意该过程并不是生成噪声beta_t并加到原图像（噪声图像）上，而是根据经过beta_t缩放的原图像和一个根据beta_t和标准正态分布采样出的噪声I得到的一个高斯分布，在从里面采样直接得到下一步噪声图像，而且beta_t是一个根据步数变化的数值（就像学习率一样），所以beta_t如何调度会明显影响收敛速度质量，原DDPM和SD 1都是线性变化的，但是后续研究证明线性调度会导致还是清晰图像时信息丢失太快，如果使用开头慢中间快结尾慢的形式，会对质量有较大影响，比如OpenAI的[Improved DDPM](https://arxiv.org/abs/2102.09672)就是使用了余弦调度，显著提升了效果，后续主流的一些图像生成的调度方式都不太一样，不过总的来说都是用的余弦相关，且需要精心设计

<div align="center">

$$ q(x_t | x_{t-1}) = N(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I) $$
$$ q(x_t | x_0) = N \big( x_t; \sqrt{\overline{\alpha}_t} x_0, (1 - \overline{\alpha}_t) I \big) $$
$$ \overline{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s) $$

</div>

- DDPM逆向生成则是训练扩散模型的过程，通常我们是让模型去预测每一步的噪声epsilon（loss常用mse，这里有个疑问，这个噪音就是正向扩散中从标准正态分布采样的噪声I），以下式1是训练loss的原始形式，通常使用的时候我们需要把x_t写成x_0的形式，在SD 1中又加入了一个平衡权重，就得到了式2，在推理过程中，也是构造一个高斯分布（式3）从中采样，其中I采样自标准正态分布，式4是均值，式5, 6都是方差系数，原文说这是两种选择，而且得到的结果都差不多，式5是直接使用前向过程的噪声方差这种做法非常直观：既然正向过程在第t步加的是方差为beta_t的高斯噪声，逆向去噪时也简单地用同样的方差进行采样，式6是使用后验分布的方差，这是严格地从贝叶斯意义上得到的方差，因为式5比较简单，所以实现上基本都用的式5，不过后续主流模型使用的采样方法五花八门各不相同，比如DPMSolver++、Euler、Flux专用的FlowMatchEuler等

<div align="center">

$$ L = \sum_{t, \epsilon \sim N(0, I)} \big[ || \epsilon - \epsilon_{\theta}(x_t, t) ||^2 \big] $$
$$ L = \frac{1 - \overline{\alpha}_t}{\beta_t} || \epsilon - \epsilon_{\theta}(\sqrt{\overline{\alpha}_t} x_0 + \sqrt{1 - \overline{\alpha}_t} \epsilon, t) ||^2, \epsilon \sim N(0, I) $$
$$ p_{\theta}(x_{t - 1} | x_t) = N(x_{t - 1}; \mu_{\theta}(x_t, t), \sigma_t I) $$
$$ \mu_{\theta}(x_t, t) = \frac{1}{\sqrt{1 - \beta_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \overline{\alpha}_t}} \epsilon_{\theta}(x_t, t)) $$
$$ \sigma_t^2 = \beta_t $$
$$ \sigma_t^2 = \beta_t \frac{1 - \overline{\alpha}_{t - 1}}{1 - \overline{\alpha}_t} $$

</div>

- SD 1的模型结构和原U-Net差别还是很大，首先是要对时间步进行编码（没有看到具体用了什么编码方式，GPT说的是Sinusoidal Positional Embedding，也就是原transformer论文中最原始那个PE，极有可能就是这个，因为SD 3就是用的这个），然后需要针对condition进行建模，用的cross attention（就是KV来自condition，Q来自latent space的z或是diffusion model中间层的输出，需要将图像的HW维（空间维度） flatten，然后要保证condition的特征维度和图像一样，不然不能做QK的矩阵乘法，空间维（对应text就是时间维）不用一样，反正Q(K^T)V做完矩阵乘法，最后得到的结果就是HW，其他和多头注意力没区别）+ feedforward net，他原文说自己就是在[Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233)的结构上增加了 a position-wise MLP 和 a cross-attention layer ，Diffusion Models Beat GANs 似乎又是改进自DDPM的网络结构，***溯源还是得去看代码！***
- SDXL 的改进主要是调整了模型结构，比如加入了两个text conditional embedding，会将他们的结构concatenate在一起带入cross attention，在输入中加入 Pooled text emb、Micro-Conditioning，做了Multi-Aspect Training适应不同的图片尺寸，改进了VAE通过维护了一个EMA权重 (exponential moving average)来作为最终VAE模型权重，还有一个用于更清晰情况下去噪的Refinement Stage Model，训练目标方法等没有做调整，整个训练是分阶段进行的，就像LLM那样，***还是得去看源码！***
- SD 3 有三个 text encoder，且是用的预训练模型，autoencoder (VQ-GAN) 也是提前训练好的，训练扩散模型时，text encoder 和 autoencoder 是冻结参数了的
- 直接看SD 3论文要搞清楚Rectified Flow Loss不太容易，问了GPT，看了HuggingFace论坛的讨论，loss应该是以下形式，Rectified Flow认为噪声图像x_t是原图像x_0和完全噪声epsilon组成的线性插值，扩散模型在这里面不是要去预测噪声，而是要去预测加噪的方向（也就是x_t相对t的倒数，epsilon - x_0），这个损失叫 Conditional Flow Matching Loss, U(0, 1)指0-1的均匀分布，SD 3还认为信噪比 SNR 在1:1也就是中间位置的时候是最难预测的，所以他们给了loss权重（所以以下loss还需要乘以该权重w），***得看源码确认！***

<div align="center">

$$ L_{CFM}(\theta) = \sum_{t \sim U(0, 1), x_0 \sim p_{data}, \epsilon \sim N(0, I)} \big[ || v_{\theta}(x_t, t) - (\epsilon - x_0) ||^2 \big] $$
$$ x_t = (1 - t) x_0 + t \epsilon $$
$$ w = \frac{t}{1 - t} \pi^{\prime} $$
$$ \pi^{\prime} = \frac{1}{s \sqrt{2 \pi}} \frac{1}{t(1 - t)} exp \big( -\frac{(log \frac{t}{1 - t} - m)^2}{2s^2} \big) $$

</div>

<p align = "center">
<img src=/img/sd3_entirety.png width="1000" />
<img src=/img/sd3_dit.png width="1000" />
</p>

- SD 3说它的 architecture builds upon the [DiT](https://arxiv.org/abs/2212.09748)，但它实际上比DiT复杂多了（SD 3是上，DiT是下），DiT的condition是用 a modification of the adaLN（adaLN, Adaptive Layer Norm 就是将条件condition融入到Layer Norm里的一种方法，就是用一个线性变换或者MLP将condition预测成Layer Norm里的缩放因子gamma和偏置beta，注意和原Layer Norm一样这都是向量哈，然后DiT又加了一个预测alpha缩放因子在FFN之后再缩放，也是向量） 加入的，而SD 3是直接和图像concatenate的，在融合text encoding时，都是取的倒数第二层的输出，对两个CLIP的输出会先连接特征维，并padding特征维，然后再与T5的序列维连接，在两个CLIP与时间步合并时SD 3会对text encoder的输出进行pooled并与时间步连接，pooled不知道是怎么操作的，***看代码！***
- SD 3使用1: 1的原提示词 (caption)和合成提示词，遵循 DALL-E 3 的合成提示词方法，并使用 vision-language model [CogVLM](https://arxiv.org/abs/2311.03079) 作为生成模型
- SD 3数据清洗：1.用 NSFW-detection models 去清除成人内容，2.有一个 rating systems 给图片打分，去掉低分图片，3.用一个 cluster-based deduplication method 去除 perceptual and semantic duplicate（详见原文附录）
- SD 3使用 2d positional frequency embeddings（不知道是什么，感觉就是原PE改成2D），使用分桶采样针对不同的图片尺寸，根据分辨率不同调整了添加噪声的强度，高分辨率下噪声添加强度会增加（实际除了基础分辨率512 * 512，其他都是添加了一个固定系数，等于3），使用 [DDPO (Diffusion DPO)](https://arxiv.org/abs/2311.12908) 进行微调让模型更好的服从提示词（详见DDPO原文）

## <span id="202502021753"> Movie Gen </span>
- Meta, 2024.10
- Movie Gen: A Cast of Media Foundation Models

## <span id="202502021754"> HunyuanVideo </span>
- Tencent Hunyuan, 2024.12
- HunyuanVideo: A Systematic Framework For Large Video Generative Models

## <span id="202503021548"> Stanford Town </span>
- Stanford University, 2023.4
- Generative Agents: Interactive Simulacra of Human Behavior

<p align = "center">
<img src=/img/stanfordtown.png width="1000" />
</p>

- [代码地址](https://github.com/joonspk-research/generative_agents)，这篇文章是做了一个实验，用LLM接入一个小社会（沙盒仿真游戏，像模拟人生那样），操作里面的人的一切行为，希望能模拟出可信的人类行为，这是一个很早期的实验，那个时候还没有LLM agent，所以整个实验结论还是令人惊喜的，且个人认为整个项目可以作为创造模拟游戏的一个很好的借鉴
- 整个实验生成了一个可交互的沙盒环境（使用[Phaser web game development framework](https://phaser.io/)），比如里面的咖啡机可以煮咖啡，床可以睡人，没有理解具体是如何将大语言模型生成的动作描述和环境里的具体指令结合起来的，可能是字段匹配，如果是字段匹配，可能就难以用于更复杂的仿真了，个人感觉是不是可以像toolformer那样教LLM用些特殊指令执行动作，或者是在prompt里面预先规定好很多特殊字符用于执行，或者是直接训练LLM agent
- LLM需要给出动作指令描述，比如为客户煮咖啡，或者是去公园散步，或者是与某人聊天，聊天需要额外开启一个会话窗口，就像平时人与LLM聊天那样聊，做其他动作的话需要LLM分步骤生成做的详细内容（当然也是额外会话，不会作为action影响沙盒，相当于就是比如你要写书，具体是什么内容，一步一步分别要写什么，慢慢细化），环境会总结人物目之所及的状态信息，并作为prompt给LLM
- 建立了一个数据库作为人物的记忆流，由LLM为记忆打分，比如分手这种记忆应该给最高分，日常生活流水账低分，同时需要有一个额外的模型抽取这些记忆的embedding，并在人物要执行动作时与当前行为对话等算余弦相似度，抽取相关的记忆作为prompt，这显然是为了应付短的上下文能力，所以或许之后兴起的LLM知识库技术也是借鉴了该论文
- 当记忆分数之和高于阈值时，会让LLM反思这些记忆，并总结抽象出更高级的记忆（我觉得这是一个很好的想法，可以用来定期总结并缩短记忆，有时候缩短记忆并不只是因为模型上下文长度限制，也是因为需要储存过长的kv cache，和需要kv做矩阵乘法），每次LLM生成动作时是给出一个plan，按plan依次执行，这是为了保证前后的一致性，避免LLM在一中午的时间里去吃三次饭（这值得注意，比如要建立一个与文章相似的仿真游戏，可能也会遇到前后动作一致性的问题，或许也要用到plan的方式，还是说在prompt里做一些额外的提示，比如：“刚刚吃过饭后，你现在决定要去干什么”）

## <span id="202503021530"> QAT </span>
- [blog](https://pytorch.org/torchtune/main/tutorials/qat_finetune.html)
- QAT是一种在模型训练过程中就考虑量化效应的方法，其主要思想在于让网络在训练时模拟低精度运算，从而提高量化后模型的性能
- 在forward中就模拟量化（近似量化后的值），让网络在前向传播时体验低精度计算的影响
- 在反向传播中近似梯度，由于量化操作通常是非可微的，所以可以用STE近似

## <span id="202506020019"> Knowledge Review </span>
- 这里要先回顾一下适用于强化学习的贝尔曼方程 ([Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation))，写的不太严谨，旨在传达意思

<div align="center">

$$ Q_t = r + V_{t + 1} $$

</div>

- 上式表示当前状态的状态动作价值 Q_t 等于奖励 r 加上下一个状态的状态价值 V_(t + 1)，该式子是一个一般规律，在所有强化学习算法中都适用
- 撇开 model-base, model-free, on-policy, off-policy 等其他方面不谈，各强化学习算法的不同就在于对状态价值 V 的定义，一般情况下，如果不单独提，都是默认状态价值等于状态动作价值的期望（注意期望是概率乘以数值之和）, 如下所示

<div align="center">

$$ V_t = E(Q_t) $$

</div>

- 三大 V 不同定义的典型就是 DQN, DDPG, SAC
- DQN继承自q-learning，因为总是假设能执行最优动作，所以认为状态价值等于最大的状态动作价值，如下所示

<div align="center">

$$ V_t = \mathcal{Max} (Q_t) $$

</div>

- DDPG 因为是确定性动作，也就是一个状态只有一个动作可执行，甚至都摒弃了‘概率’，所以认为状态价值等于状态动作价值，如下所示

<div align="center">

$$ V_t = Q_t $$

</div>

- SAC 因为加入了熵到优化目标中，所以认为状态价值等于状态动作价值与状态熵 H 之和的期望，等价于状态动作价值期望与状态熵期望之和，如下所示

<div align="center">

$$ V_t = E(Q_t + H_t) = E(Q_t) + E(H_t) $$

</div>

## <span id="202505262258"> DQN </span>
- DeepMind, 2013.12
- Playing Atari with Deep Reinforcement Learning
- 2年后 2015.2 发了 Nature, Human-level control through deep reinforcement learning
- DQN 简单来说有点像是把强化学习算法（非深度强化学习算法）的 q-learning 的值估计器改成了神经网络，在更新神经网络权重时使用经验回放机制 (experience replay, memory replay)，DQN的loss如下，其中s'是下一个时间步的状态，theta_old表示旧的Q估计器参数（这本来是target net，但是是不直接训练他的，会定期更新成actor net的参数，更新方式可以是软更新的方式 Polyak update），max(Q_theta_old(s'))表示取下一个时间步所有动作的Q预测值的最大值

<div align="center">

$$ L(\theta) = E_{(s, a, r, s^{\prime}) \sim U(D)} \big[ \big( r + \gamma max(Q_{\theta_{old}}(s^{\prime})) - Q(s | a) \big)^2 \big] $$

</div>

- 探索策略还是经典的带噪声的贪心策略

## <span id="202505262300"> DDPG </span>
- DeepMind, 2015.9
- Continuous control with deep reinforcement learning
- DDPG 改进自 [DPG (Deterministic Policy Gradient)](https://proceedings.mlr.press/v32/silver14.pdf), DDPG (Deep DPG)，DPG 证明了其优化目标为 stochastic policy gradient 在高斯噪声退化为零的特例或极限形式，其方差相对于 stochastic policy gradient 更低（GPT说的），DDPG的策略优化目标如下，注意 Q_phi 在这里是不参与梯度下降的，它相当于是像MSE一样的评价函数，从这个角度看DDPG真的很像训练回归模型

<div align="center">

$$ \underset{\theta}{\mathcal{Max}} {\kern 5pt} E_{s \sim D} \big[ Q_{\phi} \big( s, \mu_{\theta}(s) \big) \big] $$

</div>

- DDPG也借鉴了在DQN中被验证有效的经验回放机制和分离的的行动网络与目标网络，从实现的角度看DDPG真的很像 DQN 的 Actor Critic 版本，行动网络 (actor net) 是由 actor 和 critic 两个模型组成的，目标网络 (target net) 也是，训练时先由目标网络的actor选出next state的动作，再由目标网络的critic评估，然后得到行动网络的critic的优化目标去优化行动网络的critic，然后再用行动网络的critic优化行动网络的actor，然后再像 DQN 一样定期copy行动网络模型参数给目标网络（这里是必使用软更新，copy指软更新）

## <span id="202505272217"> Double DQN </span>
- DeepMind, 2015.9
- Deep Reinforcement Learning with Double Q-learning
- DQN 有一个缺陷就是预测的 Q 值会无限 overestimate, Double DQN 就是在解决该问题，Double DQN 与原 DQN 唯一的区别就是替换 max(Q_theta_old(s')) 为 Q_theta_old(s', argmax(Q_theta(s'))), 也就是原本 next state 的动作（Q值最大的动作）是 Q_theta_old 自己选出的，现在由 Q_theta 选出，但是却由 Q_theta_old 估值

## <span id="202505262304"> TD3 </span>
- McGill University, 2018.2
- Addressing Function Approximation Error in Actor-Critic Methods
- TD3 是对 DDPG 的改进，TD3 又叫 Twin Delayed DDPG，DDPG 存在以下问题：1.对超参数十分敏感，2.和 DQN 一样会 overestimate Q 值，3.Q值预测会有 incorrect sharp peak for some actions, the policy will quickly exploit that peak and then have brittle or incorrect behavior
- TD3 的改进也是三点：1. Clipped Double-Q Learning, 2. “Delayed” Policy Updates, 3. Target Policy Smoothing, 1、2都是为了解决问题2（问题1没解决），3是为了解决问题3，1是用了两个critic（行动网络和目标网络都是，当然实际实现还可以加量），在预测next state q value时用两个（目标网络的）critic中相对较小的那个，训练critic两个正常分开训练，评价（行动网络的）动作价值固定用第一个critic，2是训练actor参数次数小于训练critic，比如训练2次critic才训练一次actor（当然更新目标网络参数也和训练actor的频率一致了），3是在预测next state q value时，在目标网络actor选择动作时加入了噪声，这是源于一个假设认为相似的动作应该赋予相近的 q 值，这样就起到了平滑作用

## <span id="202506072104"> soft Q-learning </span>
- UC Berkeley, 2017.2
- Reinforcement Learning with Deep Energy-Based Policies

## <span id="202505262308"> SAC </span>
- UC Berkeley, 2018.1, 2018.12
- Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
- Soft Actor-Critic Algorithms and Applications
- SAC 作者是发了两篇论文的，第二篇相当于是对原算法的小改进，不过似乎会有人将第一篇的算法称为 SAC I，第二篇为 SAC II
- SAC 也是像 DDPG 一样的 off-policy actor critic，但是SAC是类似A2C一样的 stochastic policy 而不是 DDPG 类似的 deterministic policy，SAC不是用的DDPG, A2C类似的 [policy iteration 理论](https://en.wikipedia.org/wiki/Markov_decision_process#Policy_iteration)，他用的是 soft policy iteration 理论，不同于强化学习常用的最大化累积回报优化目标，它的优化目标是 [maximum entropy objective](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)（式1），H是熵，最终的策略优化目标如下（式2），但是因为连续动作空间的动作是无限的，所以不能直接求出期望，所以不管是TD error还是策略梯度，实际实现都是使用的蒙特卡洛无偏估计

<div align="center">

$$ \underset{\pi}{\mathcal{Max}} {\kern 5pt} \sum_t E_{(s_t, a_t) \sim \rho_{\pi}} \big[ r(s_t, a_t) + \alpha \mathcal{H} \big( \pi(s_t) \big) \big] $$
$$ \underset{\theta}{\mathcal{Max}} {\kern 5pt} E [\underset{j = 1, 2}{min} {\kern 5pt} Q_{\phi_j}(s, a) - \alpha log \pi_{\theta}(a | s)] $$

</div>

- SAC也用了Clipped Double-Q Learning，从上式可以看出，和TD3是一样的，SAC或许借鉴了Double DQN的思想，只有一个actor，就是没有目标网络的actor，所以选next state的action时是由行动网络的actor采样得到（注意SAC认为状态价值等于状态动作价值期望与状态熵期望之和，这里的采样正是状态动作价值‘期望’的体现），除以上及熵、采样之外SAC与DDPG无区别，SAC的熵不是直接使用像PPO一样的 Shannon entropy（香农熵），而是使用香农熵的蒙特卡洛无偏估计（连续差分熵），就是直接取动作概率对数的负值（这也正是状态熵‘期望’的体现）
- SAC (SAC I) 在发布后存在熵loss的系数alpha难调的困境，所以又发了一篇论文 (SAC II)，这篇文章把系数alpha改为可自动调整，公式如下（省略了期望E），注意当前的熵值 -log(a | s) 不带梯度，alpha在用于策略优化目标，critic目标函数时也不带梯度，以下公式可以理解成让当前熵逼近目标熵（超参数），大于就降低系数，小于就增大系数，大部分框架都设置了一个目标熵的默认阈值等于 - action_dim，这是一个经实践验证较优的经验性设置，***但是不知道怎么来的***（离散熵为非负数，微分熵可以为负），如果是离散动作空间，则应该考虑设置成 log(A) * coef，A为可选动作数量，coef为超参数，log(A) 等于均匀分布下的对应香农熵（均匀分布的熵值最大）

<div align="center">

$$ \underset{\alpha}{\mathcal{Min}} {\kern 5pt} \alpha [ -log \pi(a | s) - \mathcal{H}_{target}] $$

</div>

- DDPG, TD3, SAC 原本都是针对连续动作空间的环境，但SAC也有离散的版本，叫作 DSAC (Discrete SAC)，可以参考[论文](https://arxiv.org/abs/1910.07207)，DSAC与SAC的区别主要是区分离散动作空间和连续动作空间的特性，因为离散动作空间是有限且可枚举的，DSAC就没有进行蒙特卡洛无偏估计，不管是估计next state的V'还是估计策略梯度（策略梯度的优化目标其实可以看成‘当前状态’的状态价值V）以及任意状态的熵H（这次就真的是是像PPO一样的 Shannon entropy），都是给出的真正的期望版本（与各动作的概率相乘）
- 有人提出DSAC存在容易低估Q值，且熵数值变化剧烈（导致数值不稳定）的问题，特别是在高维离散动作的情况下，后又诞生 SDSAC (SD-SAC, Stable Discrete SAC)，参考[论文](https://arxiv.org/abs/2209.10081)，为了解决以上两个问题，SDSAC提出 Entropy-Penalty 和 Double Average Q-learning with Q-clip，Entropy-Penalty（式1）是在原策略梯度目标上增加惩罚项，迫使更新后的策略的熵不要太偏离原策略，Double Average Q-learning with Q-clip（式 2, 3）作者认为平均可以减轻q值低估带来的偏差，clip的灵感诞生于 PPO 的 value clipping，用以防止critic的大幅更新

<div align="center">

$$ \underset{\phi}{\mathcal{Max}} {\kern 5pt} \mathcal{J}_{\pi}(\phi) = E_{s_t \sim D} [E_{a_t \sim \pi_{\phi}} [Q_{\theta}(s_t, a_t) - \alpha log \pi_{\phi}(a_t | s_t)]] - \beta \frac{1}{2} E_{s_t \sim D} (E_{a_t \sim \pi_{\phi_{old}}}[-log \pi_{\phi_{old}}(a_t | s_t)] - E_{a_t \sim \pi_{\phi}}[-log \pi_{\phi}(a_t | s_t)])^2 $$
$$ y = r + \gamma E_{a^{\prime} \sim \pi}[\mathcal{avg}(Q_{\theta_1^{\prime}}(s^{\prime}, a^{\prime}), Q_{\theta_2^{\prime}}(s^{\prime}, a^{\prime}))] $$
$$ \underset{\theta_i}{\mathcal{Min}} {\kern 5pt} \mathcal{L}_{\theta_i} = max((Q_{\theta_i} - y)^2, (Q_{\theta_i^{\prime}} + clip(Q_{\theta_i} - Q_{\theta_i^{\prime}}, -c, c) - y)^2) $$

</div>

## <span id="202506072111"> C51 </span>
- DeepMind, 2017.7
- A Distributional Perspective on Reinforcement Learning

## <span id="202506072114"> QR-DQN </span>
- DeepMind, 2017.10
- Distributional Reinforcement Learning with Quantile Regression

## <span id="202506072135"> D4PG </span>
- DeepMind, 2018.4
- Distributed Distributional Deterministic Policy Gradients

## <span id="202506072125"> IQN </span>
- DeepMind, 2018.6
- Implicit Quantile Networks for Distributional Reinforcement Learning

## <span id="202506071230"> Distributional-SAC </span>
- Tsinghua University, 2020.1, 2023.10
- Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors
- Distributional Soft Actor-Critic with Three Refinements
- 第一篇论文的算法又被称为 DSAC or DSACv1（和离散SAC重名），第二篇论文的算法又称作 DSAC-T or DSACv2

## <span id="202506072143"> Diffusion-QL </span>
- Twitter, 2022.8
- Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning

## <span id="202506072148"> Decision Diffuser </span>
- MIT, 2022.11
- Is Conditional Generative Modeling all you need for Decision-Making?

## <span id="202506072151"> Diffusion Policy </span>
- Columbia University, 2023.3
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

## <span id="202506032314"> DACER </span>
- Tsinghua University, 2024.5, 2025.5
- Diffusion Actor-Critic with Entropy Regulator
- Enhanced DACER Algorithm with High Diffusion Efficiency
- 第一篇文章提出的算法称之为 DACER, DACER是off-policy actor critic算法，DACER的特色在于：1.使用了DDPM的reverse diffusion process（逆向生成过程，提示以上SD1也使用的是DDPM的正向扩散和逆向生成，可以去参考详细公式）作为actor生成action的过程，作者认为diffusion模型比传统常用于连续动作的高斯分布能更好地建模复杂的多模态行为，2.有行动网络和目标网络的critic，但只有一个actor，且使用了两个模型组成critic（评估时选小的），和SAC一样，但是为了解决Q-value overestimation，用了 tricks in Distributional-SAC，3.加了DDPG类似的探索策略（注意是额外加入，diffusion生成本来就带有随机采样，相当于是在此基础上再加一个DDPG那种噪声），探索策略（噪声）带系数且是动态确认的，确认的方法和SAC的熵系数alpha是一样的，但是因为diffusion策略的熵没有解析解，所以作者用混合高斯模型 (Gaussian mixture model, GMM) 的熵代替（相当于认为原policy就是混合高斯模型，拟合出来，再求熵），混合高斯模型会通过policy采样n个动作出来并用 EM (Expectation-Maximization) 算法拟合（注意通过这种方式拟合的分布的熵是没有梯度的，可以直接调用sklearn）
- 感觉DACER是不是有计算复杂度太高的问题，因为他生成每一个动作都需要diffusion采样多步（当然backward的时候还要要求梯度回传多步），能理解作者为什么会加个额外探索项，因为diffusion的探索强度（熵没有解析解，导致不能放进优化目标和loss里）不能直接优化，用diffusion作为policy还有一个麻烦是难以做deterministic action，因为diffusion model做动作时就会采样，作者是通过固定随机数种子的方式让采样结果相同（作者代码是用JAX写的，看了半天才看懂...），但是这肯定是一种伪确定性方式
- 第二篇文章称为 DACER2，DACER2的创新点主要就是加入了 Langevin Dynamics，**整体都没看懂，现在还没开源代码，好在DACER的issue里面说这次要开源pytorch代码，等等看**，有一个很大的疑问就是关于S_theta(s, a, t)，和GPT讨论了很久也没讨论清楚（和GPT各自都不能说服对方...），我认为既然新增了一个Loss是关于Q梯度的和S_theta的，那S_theta肯定是一个待优化的新模型，需要预测Q梯度并在反向生成阶段用来去噪，GPT说的S_theta就是actor...按GPT的说法相当于actor除了输出action还得承担接收action, state, t并预测Q梯度的任务，显然不现实，但是我这样想又是违背公式12的，公式12里直接将Q梯度实际值加在反向生成去噪过程中了，但是既然要这样加，为什么又要在策略梯度里加一个以Q梯度为目标的loss，另外作者还为Q梯度loss加了和时间相关的权重，去噪初期权重大，后期小，让初期大力朝最大化Q值方向移动，去掉‘大噪声’，后期则更关注另一个loss（关键是另一个loss明明也是-q，也是最大化Q值，人都麻了）