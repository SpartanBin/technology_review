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
- [Flamingo (DeepMind, 2022.4)](#202502151055)
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
- [U-Net (University of Freiburg, Germany, 2015.5)](#202503051357)
- [Stable Diffusion (CompVis, Runway ML, 2021.12)](#202503051346)
- [Movie Gen (Meta, 2024.10)](#202502021753)
- [HunyuanVideo (Tencent Hunyuan, 2024.12)](#202502021754)
- [Stanford Town (Stanford University, 2023.4)](#202503021548)
- [QAT](#202503021530)

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

- Llama 2使用了两种RL的方法，一种就是PPO，另一种是Rejection Sampling fine-tuning（RS），在前几轮里只用RS，后面把PPO用到RS的最优结果上，然后他们最大的模型才用了RS，小的模型都是使用的大模型RS的样本（他们称此为蒸馏 distillation）
- Rejection Sampling fine-tuning（RS）就是对一个prompt生成多个回答，然后选出RM评分最高的那个，然后用最大化对数概率之和训练（这个是我猜的，因为他毕竟叫fine-tuning，原文又没写）
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

## <span id="202502021745"> DALL-E 1 2 3 </span>
- OpenAI

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

## <span id="202503051357"> U-Net </span>
- University of Freiburg, Germany, 2015.5
- U-Net: Convolutional Networks for Biomedical Image Segmentation

<p align = "center">
<img src=/img/unet_backborn.png width="1000" />
<img src=/img/unet_overlaptile.png width="1000" />
</p>

- 上图都画得很清楚，模型有编码器(Contracting Path)和解码器(Expansive Path)组成，最后输出层是(1, 1)卷积，全模型没使用linear transformation，编码器的每个块由2个(3, 3)的卷积（无padding和stride）和一个(2, 2)的max pooling组成，解码器的每个块由1个(2, 2)的反卷积（上采样）和2个(3, 3)的卷积（无padding和stride）组成，另外因为是用来做医学图像分割，所以最后输出的通道是2（2分类任务）
- 解码器中每个块上采样完之后，会与编码器中对应的块的卷积结果（先裁剪4边到适合解码器的尺寸(crop)）concatenate到一起，这样做论文说是为了加强高分辨率细节
- 看上图1可以看出最后输出比输入的长宽小，所以论文利用overlap tile就是把分割图像先分割成重叠的若干份，最后再拼接起来，如果是边界位置就用镜像扩展，这样保证了与label的对应关系（因为label与输入尺寸一样）
- 训练的时候加了个weight map，用来提到细胞间分界线这些像素在loss里面的权重，用来让模型更关注细胞边界

## <span id="202503051346"> Stable Diffusion </span>
- CompVis, Runway ML, 2021.12
- High-Resolution Image Synthesis with Latent Diffusion Models

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