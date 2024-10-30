# HK-O1-Law

[包括模型介绍，参数，预期的场景@Sirui Han，讲一讲背景]
6 ~ 7 行

## Examples

放 3 - 5 典型的case

# Dataset Overview

O1aw-Dataset is a comprehensive legal question-thought-answer dataset, designed to evaluate and enhance legal reasoning capabilities in language models. The dataset follows the O1-style format, featuring complex legal scenarios that require multi-step reasoning.

## Our training dataset
How was our dataset constructed? First, we crawled and cleaned raw legal materials from the internet, including [Hong Kong e-Legislation](https://www.elegislation.gov.hk). Then, we used GPT-4o to generate corresponding questions and thought-answer pairs based on the raw legal materials.

The dataset contains 15,959 question-thought-answer triples, each equipped with complete chain-of-thought annotations. All content is presented in Simplified Chinese and stored in a structured JSON format. The difficulty level of the questions in this dataset is intermediate to advanced for legal professionals and law school students.

The question types cover case analysis, legal application, explanation of legal concepts and so on. Each QTA triple includes detailed question prompt, a 3-5 step chain-of-thought reasoning process, and answer. The reasoning process involves multi-stage validation, reflective verification steps, and cross-case consistency checks, ensuring diversity in reasoning.

### Prompts for QTA generation
Here is our prompt template for Question generation:
```python
SYSTEM_PROMPT: str = """
# Task
基于以下参考法律材料，生成至少{n}个法律问题。问题需要满足以下要求：

1. 复杂度要求：
- 问题的答案需要通过对参考法律材料的深入理解和分析才能得出
- 应该是一个开放性问题，需要进行推理和解释
- 不能是简单的是非题或事实性问题

2. 问题形式：
- 可以是案例分析题
- 可以是法律适用题
- 可以是法律概念解释题

3. 问题结构：
- 明确的问题陈述

4. 难度级别：
中等难度，适合法律专业学生或从业者思考和讨论


5. 输出格式：
请严格使用JSON格式输出，结构如下：
{{
  "questions": [
    {{
      "id": 1,
      "type": "案例分析/法律适用/概念解释...", 
      "question": "具体问题",
    }},
    {{
      "id": 2,
      ...
    }},
    {{
      "id": 3,
      ...
    }}
  ]
}}
"""

USER_PROMPT: str = """
# 参考法律材料
{prompt}

# 提示
生成的问题应该与提供的参考法律材料直接相关，但是必须假装参考法律材料对你不可见。
请以JSON格式输出：
"""
```

Here is our prompt template for Thought(COT) and Answer generation:
```python
SYSTEM_PROMPT: str = """你是一个专家级的AI助手，能够逐步解释推理过程。你将收到一个问题和相关参考资料。你的任务是重构并展示通向正确答案的完整推理路径。

对于每个推理步骤，提供一个标题，描述你在该步骤中所做的事情，以及内容。但必须展示至少三种不同的方法或途径来得出该答案。

要求：
1. 使用3-5个推理步骤
2. 探索多种方法以达到答案
3. 通过不同的方法验证给定答案
4. 考虑潜在的替代答案并解释为何被拒绝
5. 你必须假装没有参考资料，只可以把参考资料当作自己的知识
6. 考虑你可能是错的，如果你的推理是错的，它会在哪里
7. 充分测试所有其他可能性。你可能会错
8. 当你说你正在重新检查时，请真正重新检查，并使用另一种方法进行，不要只是说你正在重新检查

以JSON格式回应，包含以下键：
- 'title': 当前推理步骤的描述
- 'content': 该步骤的详细解释
- 'next_action': 'continue' 或 'final_answer'
有效的JSON响应示例：
[
  {{ 
      "title": "分析给定信息", 
      "content": "首先，让我们检查问题，以识别将指导我们解决过程的关键要素……", 
      "next_action": "continue"
  }},
  {{ 
      "title": "...", 
      "content": "...", 
      "next_action": "continue"
  }},
  ...
  {{ 
      "title": "...", 
      "content": "...", 
      "next_action": "final_answer"
  }}
]
"""

USER_PROMPT: str = """
# 问题：
{prompt}
# 参考资料：
{references}

请以JSON格式输出：
"""

```


## Training details

We use [Align-Anything](https://github.com/PKU-Alignment/align-anything) framework to conduct SFT training on [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). The training dataset and hyperparameters used are detailed below.

### Chat Template

Specifically, in SFT training, Q, T, and A in the [training dataset](https://huggingface.co/datasets/HKAIR-Lab/O1aw-sft-16k) are segmented by reserved tokens in the tokenizer of Llama-3.1-8B. For more details, see template:

```
system_prompt: str = ''
user_prompt: str = '<|reserved_special_token_0|>{input}<|reserved_special_token_1|>\n'
assistant_thinking: str = '<|reserved_special_token_2|>{thinking}<|reserved_special_token_3|>\n'
assistant_answer: str = '<|reserved_special_token_4|>{answer}<|reserved_special_token_5|>'
template = system_prompt + user_prompt + assistant_thinking + assistant_answer
```

### Hyperparameters

train_cfgs

    ```
    # The deepspeed configuration
    ds_cfgs: ds_z3_config.json
    # Number of training epochs
    epochs: 3
    # Seed for random number generator
    seed: 42
    # Batch size per device for training
    per_device_train_batch_size: 4
    # Batch size per device for evaluation
    per_device_eval_batch_size: 4
    # The number of gradient accumulation steps
    gradient_accumulation_steps: 16
    # Whether to use gradient checkpointing
    gradient_checkpointing: True
    # Initial learning rate
    learning_rate: 2.e-5
    # Type of learning rate scheduler
    lr_scheduler_type: cosine
    # Ratio of warmup steps for learning rate
    lr_warmup_ratio: 0.03
    # Weight decay coefficient
    weight_decay: 0.0
    # Hyper-parameters for adam optimizer
    adam_betas: [0.9, 0.95]
    # Hyper-parameters for adam epsilon
    adam_epsilon: 1.e-8
    # Enable bfloat 16 precision
    bf16: True
    # Enable float 16 precision
    fp16: False
    # The strategy of evaluation, choosing form [epoch, steps]
    eval_strategy: epoch
    # The evaluation interval in step-wise evaluation case
    eval_interval: 10
    # The max norm of gradient
    max_grad_norm: 1.0
    ```

data_cfgs

    ```
    # Datasets to use for training
    train_datasets: HKAIR-Lab/O1aw-sft-15k
    # The split of train datasets
    train_split: train
    ```

model_cfgs

    ```
    # Pretrained model name or path
    model_name_or_path: meta-llama/Llama-3.1-8B 
    # Whether to trust remote code
    trust_remote_code: True
    # The max token length
    model_max_length: 2048
    ```

For other hyperparameters, we use the default value in the [configure file](https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/configs/train/text_to_text/sft.yaml). 

## Evaluation for HK Laws

In the planning, coming soon.


## Our teams


## Citation
Please cite the repo if you use the data or code in this repo.

```bibtex
@misc{align_anything,
  author = {HKAIR Lab},
  title = {HK-O1-Law Models: A HK Law Large Language Model using O1's Slow Thinking},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HKAIR-Lab/HK-O1-Law}},
}
```


## License

HK-O1-Law models and datasets are released under Apache License 2.0.
