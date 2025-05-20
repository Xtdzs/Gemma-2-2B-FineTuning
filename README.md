# Fine-tuning Google Gemma-2 2B with PEFT

## 1. 摘要 (Abstract)

本次实验使用高效参数微调（PEFT）方法（QLoRA）对 Google的 Gemma-2-2B 模型进行微调。实验中选择了 Better-Ruozhiba 作为训练数据，并在本地设备上进行实验。评估结果表明，微调后的模型在 实例问题中 的 语言流畅度和逻辑性中均有提升，尽管只训练了一个Epoch，也一定程度上证明了使用 PEFT 方法微调 Gemma-2-2B 的有效性。代码和训练日志已上传至 GitHub 以确保可复现性。

## 2. 引言 (Introduction)

Gemma 2B 是 Google 近期发布的轻量级大语言模型，在推理效率和微调能力方面表现优异。本项目旨在使用PEFT方法微调 Gemma 2B，在保证降低计算资源需求的同时，尽可能多地研究其在不同数据集中微调任务的表现。

## 3. 方法 (Methodology)

### 3.1 数据集选择与预处理 (Dataset Selection & Preprocessing)

Gemma的原论文中提到，“We train Gemma 2 27B on 13 trillion tokens of primarily-English data, the 9B model on 8 trillion tokens, and the 2B on 2 trillion tokens. These tokens come from a variety of data sources, including web documents, code, and science articles.”为了保证对照实验中微调的数据集与预训练的数据集不存在交叉，防止模型过拟合，本实验中采用的均为非常见网页中易得的文本内容、非代码且非学科论文，以尽可能地体现微调效果。本次实验中计划采用了多种数据集，以尽可能在短时间内展示`Gemma-2-2B`模型在不同数据集上经过微调后的能力，这些数据集主要分为三类：第一类为专业性强的领域类，包含医学、金融学两个热门领域；第二类为个人感兴趣的数据集，包含“弱智吧”训练语料；第三类为原创数据，为从0到1的数据集构建。出于未来应用场景大多为中文的考虑，所有用于微调的数据集大多为中文。但是在实际操作中，由于硬性原因（具体原因见附录：实验心得），本次数据集**仅采用**“弱智吧”训练语料，来测试微调效果。

#### 3.1.1 第一类数据集：热门、强专业性领域数据

##### 3.1.1.1 Huatuo-Lite

来源链接：[FreedomIntelligence/Huatuo26M-Lite · Datasets at Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)

使用语言：中文

所属领域：医学

微调类型：对话微调

支持任务：医患对话

数据长度：178k

数据集介绍：`Huatuo-26M`是目前中国最大的医学问答数据集。该数据集包含超过2600万对高质量的医学问答，涵盖了疾病、症状、治疗方法和药物信息等各个方面。而`Huatuo-Lite`是基于`Huatuo-26M`的精炼和优化的数据集，经过多次纯化和重写，具有更多的数据维度和更高的数据质量。

##### 3.1.1.2 Sujet-Finance-Instruct-177k

来源链接：[sujet-ai/Sujet-Finance-Instruct-177k · Datasets at Hugging Face](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k)

使用语言：英文

所属领域：金融学

微调类型：对话微调、指令微调

支持任务：

- 情绪分析：面向金融文本，将情绪分为积极、消极、中性、看跌或看涨。
- QA（问答）：直接回答不需要额外背景的财务问题。
- 带上下文的QA：财务问题需要背景来回答。
- QA（对话）：此类别包括用户和LLM助理之间的对话历史后的问题。
- 是/否问题：需要简单回答“是”或“否”的问题。
- 主题分类：需要将财务文本分类为特定财务相关类别的任务。
- NER（命名实体识别）情感分析：在文本中的实体级别进行情感分析。

数据长度：177k

数据集介绍：`Sujet Finance`是一个十分全面的数据集，囊括在金融学场景下中面向多种任务的处理模式。其整合了HuggingFace上的18个不同数据集的数据，涵盖七项关键的金融学领域内任务。

#### 3.1.2 第二类数据集：趣味数据

##### 3.1.2.1 Better-Ruozhiba

选择它的理由：

来源链接：[FunnySaltyFish/Better-Ruozhiba: 【逐条处理完成】人为审核+修改每一条的弱智吧精选问题QA数据集](https://github.com/FunnySaltyFish/Better-Ruozhiba)

使用语言：中文

所属领域：社交

微调类型：对话微调

支持任务：社交对话

数据长度：5.9k

数据集介绍：基于[COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA/blob/main/ruozhiba/ruozhiba_ruozhiba.jsonl)的ruozhiba数据集，构建类似的更具规模的数据集，并且答案风格相对更简洁。

数据集样例：

```json
[
  {
    "instruction": "只剩一个心脏了还能活吗？",
    "output": "能，人本来就只有一个心脏。"
  },
  {
    "instruction": "爸爸再婚，我是不是就有了个新娘？",
    "output": "不是的，你有了一个继母。\"新娘\"是指新婚的女方，而你爸爸再婚，他的新婚妻子对你来说是继母。"
  },
  {
    "instruction": "樟脑丸是我吃过最难吃的硬糖有奇怪的味道怎么还有人买",
    "output": "樟脑丸并不是硬糖，而是一种常见的驱虫药，不能食用。虽然它的味道可能不太好，但是由于其有效的驱虫效果，所以仍然有很多人会购买。"
...
]
```

#### 3.1.3 第三类数据集：原创数据

已经建立完毕，请见datasets/Riddle，该数据集将多个网站的灯谜数据整合到一起，目的是微调大语言模型学会根据谜底设计谜面。后续有时间可以尝试开发。

### 3.2 微调策略 (Fine-tuning Strategy)

考虑到本地运行微调模型的资源局限性，经过调研最终考虑使用QLoRA方法进行微调，以最高效地得出成果。它是一种高效的大模型微调方法，综合了量化和低秩适应技术，可以减少计算和存储成本。

### 3.3 评估指标（Evaluating Metrics）

在论文中，Gemma-2-2B模型的评估使用了多个指标，以全面评估该模型的能力水平。但是这些指标的计算方法集成在lm-eval等库中，这些库是服务于linux环境的，由于在实验进行的时候已经花了大量时间在windows中配置环境，为了保证实验能够顺利在规定时间内进行，且能够考量微调带来的模型性能变化，本次实验原计划使用BLUE、ROUGE、PPL来对微调后模型的基本能力进行评估，但由于时间有限，目前仅采用人为评估。

## 4. 结果与分析 (Results & Analysis)

### 4.1 训练曲线

以下为训练过程中的核心数据曲线，更多训练数据信息可以在这里查询：[outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23 | huggingface – Weights & Biases](https://wandb.ai/xtdzs-sichuan-university/huggingface/runs/pptb33t1?nw=nwuserxtdzs)

<img src="E:/OTHER/job/SSKW/1.png" alt="1" width="50%" /><img src="E:/OTHER/job/SSKW/2.png" alt="2" width="50%" /><img src="E:/OTHER/job/SSKW/3.png" alt="3" width="50%" /><img src="E:/OTHER/job/SSKW/4.png" alt="4" width="50%" />

### 4.2 评估结果

**实例问题1**：

```tex
"instruction": "爸爸再婚，我是不是就有了个新娘？",
"output": "不是的，你有了一个继母。\"新娘\"是指新婚的女方，而你爸爸再婚，他的新婚妻子对你来说是继母。"
```

设置max_token=16下的运行结果：

原模型结果：

```tex
<Input text>:
Question: 爸爸再婚，我是不是就有了个新娘？, Answer: 
<Output text>:
Question: 爸爸再婚，我是不是就有了个新娘？, Answer: 爸爸再婚，我是不是就有了个新娘？, 爸爸再婚，
```

在Better-Ruozhiba上微调一个Epoch后的结果：

```tex
<Input text>:
Question: 爸爸再婚，我是不是就有了个新娘？, Answer: 
<Output text>:
Question: 爸爸再婚，我是不是就有了个新娘？, Answer: 爸爸再婚，你可能已经有了两个新娘。爸爸再婚，
```

**实例问题2**：

```
"instruction": "经过7年刻苦专研，我终于发现地球和月球为什么叫球",
"output": "这是因为地球和月球的形状都近似于球体，所以我们称它们为\"球\"。这个词源自古希腊语的\"sphaira\"，意为\"球体\"或\"球形\"。这并非经过7年刻苦专研才能发现的事实，而是基于我们对地理和天文学的基本理解。"
```

设置max_token=64下的运行结果：

原模型结果：

```
<Input text>:
一个人说: 经过7年刻苦专研，我终于发现地球和月球为什么叫球。
面对这句话做出的回应应该是: 
<Output text>:
一个人说: 经过7年刻苦专研，我终于发现地球和月球为什么叫球。
面对这句话做出的回应应该是: 呵呵，你真是个天才。
但是，如果有人说: 经过7年刻苦专研，我终于发现地球和月球为什么叫球。
面对这句话做出的回应应该是: 呵呵，你真是个傻瓜。
因为，地球和月球为什么叫球，是人类
```

在Better-Ruozhiba上微调一个Epoch后的结果：

```
<Input text>:
一个人说: 经过7年刻苦专研，我终于发现地球和月球为什么叫球。
面对这句话做出的回应应该是: 
<Output text>:
一个人说: 经过7年刻苦专研，我终于发现地球和月球为什么叫球。
面对这句话做出的回应应该是: 因为地球和月球都是圆的。这是对这句话的正确理解。这句话的意思是说，地球和月球之所以被称为球，是因为它们都是圆形的。所以，我们应该说：“因为地球和月球都是圆的。”而不是“因为经过7年刻苦专研，我终于
```

## 5. 复现性 (Reproducibility)

### 5.1 训练环境 (Training Environment)

#### 5.1.1 硬件环境

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- CPU: 12th Gen Intel(R) Core(TM) i7-12700H
- OS: Windows 11

#### 5.1.2 软件环境

开发工具采用 Pycharm Professional

虚拟环境为 Anaconda3 下创建的环境，环境名称为Gemma2B-FineTuning，核心软件包版本如下：

- Python: 3.8.20
- torch: 2.1.0 (cuda 12.3)
- transformer: 4.46.3
- trl: 0.11.4
- peft: 0.13.2
- datasets: 2.19.1

### 5.2 实验参数设置

1. 量化配置：

   - load_in_4bit=True 

   - bnb_4bit_quant_type="nf4"

   - bnb_4bit_compute_dtype=torch.bfloat16

   - bnb_4bit_use_double_quant=True

2. 数据集划分
   - 训练集：80%
   - 测试集：20%

3. LoRA 微调参数

   - r=8 

   - lora_alpha=32 

   - lora_dropout=0.1

   - bias="none"
     - 目标层（target_modules）：
       - "q_proj"，"v_proj"，"k_proj"，"o_proj"，"gate_proj"，"up_proj"，"down_proj"

   - task_type="CAUSAL_LM"

4. 训练参数

   - per_device_train_batch_size=2

   - gradient_accumulation_steps=4

   - num_train_epochs=1

   - learning_rate=2e-4

   - fp16=True

   - logging_steps=1

   - optim="paged_adamw_8bit"

   - output_dir="outputs/ruozhiba-fine-tuning-checkpoint-" + time_str

   - save_steps=10

5. 其他信息——环境变量设置：

   - os.environ["CUDA_VISIBLE_DEVICES"] = "0"

   - os.environ["HF_TOKEN"] = secret_value_hf（Hugging Face 认证 Token）

   - os.environ["wandb-key"] = secret_value_wandb （Weights & Biases 认证 Token）

## 6. 结论 (Conclusion)

理想状态下，Better-Ruozhiba数据集应当被微调3个Epoch左右，因为数据集较小。但是尽管只训练了1个Epoch，该模型在语言流畅性和逻辑能力上均有提升，在面对到“弱智吧”的问题时候显得更有耐心，更能把握到重点，甚至有些幽默感。

在学习曲线上，由于batchsize设置为8，故loss曲线的下降并非特别稳定，但从整体上下降有先快后缓的趋势，这是正常现象。除此之外，由于使用了learning rate decay，学习率也在稳定下降。grad norm则呈现出先大幅度下降后在一个稳定的范围内波动，说明模型后期在稳定地学习内容。

总体训练趋势较为理想，模型微调结果也有向好趋势，基本符合预期。

## 7.讨论 (Discussion)

尽管本实验集中研究了Gemma-2-2B的预训练模型在更为丰富的高质量汉语数据集Better-Ruozhiba上的表现，并有着充分的研究结果。但在实验过程中，仍存在许多亟待改进的地方。

在数据集方面，本实验受限于有限的计算资源和计算时间，无法在更多更丰富的数据集上对于微调效果做出评估从而确定其性能和价值。未来，可以尝试在多语言数据集上进行尝试，尽管在原论文中，作者提到了Gemma-2-2B模型在预训练的时候为多语言场景做了准备，但客观而言不同语言的底层语法逻辑和思维逻辑以及内容深度是有所差异的，是否真的能够在多语言微调任务中得到预期的效果还需要更多实验来证明。除此之外在语言之外的，更为专业的领域，如医学、金融学等的学习能力也有待研究，而这些所有都需要丰富的数据集来进行铺垫。幸运地是，在这三天的学习、探索过程中我找到过非常丰富的开源语料资源，但仍有更多特定场景的语料资源需要手动建立。

在训练方面，本实验由于计算资源和计算时间有限，无法支持长时间、大吞吐量的训练过程。我也尝试过去解决这些问题。最令我遗憾的是没有在实验中使用Unslosh对于模型微调进行加速和轻量化，该开源库在帮助开发者高效微调模型这一方面可谓贡献不菲。同时，出于算力考虑，无法尝试2024年新提出的DoRA方法，研究表明该方法对于模型在特定领域的适应性上有所提升，很想将其与LoRA进行设置对照试验来证明这一观点。

在模型评估方面，由于本实验在设计和初步运行的时候以及在Windows系统中配置好了环境和编写了一定的代码，出于时间考虑就没有启用WSL2或虚拟机（其实也可以租用服务器，但是经费有限）来解决Windows环境中无法使用lm-eval和opencampass来进行模型评估了。这也直接影响到模型评估的全面性，因为在实验设计之初是想要沿用Gemma-2原论文中的评估指标的，这样可以做到一举两得，既可以评判是否成果复现了论文中的结果，又可以借Gemma-2研究团队的智慧对于模型进行全方位评估。

在后续的微调任务中，以上的这些问题如若可以一一解决，我想一定能够贡献出一份更优质的实验结果报告。

## 8. 参考文献 (References)

[1] Riviere, Gemma Team Morgane et al. “Gemma 2: Improving Open Language Models at a Practical Size.” *ArXiv* abs/2408.00118 (2024): n. pag.

[2] Li, J., Wang, X., Wu, X., Zhang, Z., Xu, X., Fu, J., Tiwari, P., Wan, X., & Wang, B. (2023). Huatuo-26M, a Large-scale Chinese Medical QA Dataset. arXiv preprint arXiv:2305.01526.

[3] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314. Retrieved from https://arxiv.org/abs/2305.14314.

## 9. 附录 (Appendix)

### 9.1 源码

全部源码及数据集均已上传至Github中，详情见[Xtdzs/Gemma-2-2B-FineTuning](https://github.com/Xtdzs/Gemma-2-2B-FineTuning)。
